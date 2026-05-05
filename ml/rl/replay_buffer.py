"""
ml/rl/replay_buffer.py
-----------------------
Experience Replay Buffers pour DQN.

Deux implémentations :
  1. UniformReplayBuffer   : sampling uniforme (baseline)
  2. PrioritizedReplayBuffer : prioritized experience replay (PER)
     → Samples transitions with higher TD-error more frequently
     → Significantly improves convergence & stability

Implémentation PER : Segment tree pour efficacité O(log N)
  → Sans dépendances externes, pur NumPy + structure de données classique
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


class UniformReplayBuffer:
    """
    Buffer circulaire uniforme pour l'Experience Replay.
    Baseline : sampling aléatoire uniforme.

    Paramètres
    ----------
    capacity  : capacité maximale du buffer
    obs_dim   : dimension du vecteur d'observation
    """

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.size = 0
        self.ptr = 0

        # Pré-allocation numpy pour l'efficacité
        self.states      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retourne batch uniforme + weights + indices (pour compatibilité PER).
        Poids = 1/N uniforme pour tous les samples.
        Indices = positions dans le buffer (pour priority update).
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        weights = np.ones(batch_size, dtype=np.float32) / self.size
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
            weights,
            idxs,  # indices
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """No-op pour buffer uniforme (compatibility interface)."""
        pass

    def __len__(self) -> int:
        return self.size

    @property
    def is_ready(self) -> bool:
        return self.size >= 500


class SegmentTree:
    """
    Segment tree pour Prioritized Experience Replay.
    Efficacité : O(log N) pour update et sample.
    
    Structure : arbre binaire où chaque node = somme de ses enfants.
    Utilisé pour :
      1. Accumuler les priorities
      2. Échantillonner proportionnellement à la priority
    """

    def __init__(self, capacity: int):
        # Arbre : capacity éléments au dernier niveau
        # Size du tree = 2*capacity (approximation, suffisant)
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.ptr = 0

    def set(self, idx: int, value: float) -> None:
        """Set leaf at idx and propagate changes up."""
        idx += self.capacity
        delta = value - self.tree[idx]
        self.tree[idx] = value
        
        # Propagate up
        while idx > 1:
            idx //= 2
            self.tree[idx] += delta

    def get(self, idx: int) -> float:
        """Get priority at leaf idx."""
        return self.tree[idx + self.capacity]

    def sum(self) -> float:
        """Total sum (root value)."""
        return self.tree[1]

    def sample(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        """Sample batch_size indices proportional to priorities."""
        total = self.sum()
        batch = []
        segment = total / batch_size

        for i in range(batch_size):
            # Sample in [i*segment, (i+1)*segment)
            target = rng.uniform(i * segment, (i + 1) * segment)
            idx = self._search(target)
            batch.append(idx)

        return np.array(batch, dtype=np.int32)

    def _search(self, target: float) -> int:
        """Binary search pour trouver leaf où cumsum >= target."""
        idx = 1
        while idx < self.capacity:
            if self.tree[2 * idx] > target:
                idx = 2 * idx
            else:
                target -= self.tree[2 * idx]
                idx = 2 * idx + 1
        return idx - self.capacity


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER).
    
    🧠 POURQUOI PER EST CRITIQUE EN RL
    ──────────────────────────────────
    Problème standard : replay buffer uniform sampling
      ❌ Transitions faciles (low TD-error) sont ré-entraînées inutilement
      ❌ Transitions difficiles (high TD-error) sont négligées
      ❌ Convergence lente, inefficacité de l'entraînement
    
    Solution PER :
      ✅ Priorité = |TD-error| : réentraîner les erreurs graves
      ✅ Importance sampling : corriger le bias introduit par non-uniform sampling
      ✅ Résultat : +30-50% convergence speed sur DQN
      ✅ Variance réduite dans les gradients
    
    Paramètres clés :
      alpha   : 0.0 = uniform (baseline)
                1.0 = full prioritization (peut être instable)
                0.6 = bon compromis (recommandé)
      
      beta    : 0.4 → 1.0 : annealing du poids d'importance
                Au début: down-weight les samples prioritaires
                À la fin: correction complète (importance weights = 1)
                Raison: TD-errors deviennent plus stables en fin d'entraînement
      
      epsilon : 1e-6 : clipping minimal pour éviter priority=0
    """

    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.alpha = alpha  # prioritization strength
        self.beta = beta    # importance sampling weight
        self.epsilon = 1e-6

        # States storage
        self.states      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

        # Priority tree
        self.tree = SegmentTree(capacity)
        self.max_priority = 1.0
        self.ptr = 0
        self.size = 0
        
        # RNG
        self.rng = np.random.default_rng()

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition with max priority (prioritize new experiences)."""
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = float(done)

        # New transitions get max priority
        priority = self.max_priority ** self.alpha
        self.tree.set(self.ptr, priority)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample batch with PER.
        Retourne : (states, actions, rewards, next_states, dones, importance_weights, indices)
        """
        # Sample indices according to priorities
        idxs = self.tree.sample(batch_size, self.rng)

        # Compute importance sampling weights
        # IS weight = (1 / (N * P(i)))^beta
        priorities = np.array([self.tree.get(i) for i in idxs], dtype=np.float32)
        probabilities = priorities / self.tree.sum()
        
        # Importance sampling correction
        weights = (self.size * probabilities) ** (-self.beta)
        weights /= weights.max()  # Normalize [0, 1]
        weights = weights.astype(np.float32)

        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
            weights,
            idxs,  # indices for priority update
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities based on TD-errors.
        
        Priority ∝ |TD-error| : les transitions avec grosse erreur
        sont réechantillonnées plus souvent.
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (np.abs(td_error) + self.epsilon) ** self.alpha
            self.tree.set(int(idx), priority)
            self.max_priority = max(self.max_priority, priority)

    def anneal_beta(self, episode: int, total_episodes: int) -> None:
        """
        Increase beta from initial value to 1.0 during training.
        Importance sampling correction gets stronger over time.
        """
        initial_beta = 0.4
        self.beta = initial_beta + (1.0 - initial_beta) * (episode / max(total_episodes, 1))

    def __len__(self) -> int:
        return self.size

    @property
    def is_ready(self) -> bool:
        return self.size >= 500