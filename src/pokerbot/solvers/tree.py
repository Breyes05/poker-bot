"""A game tree compiled into flat numpy arrays.

The reference solvers walk the tree through the ``Game`` API on every
iteration: millions of ``next_state``/``payoff``/``is_terminal`` calls whose
answers never change. Profiling showed that this, rather than any single
function, was the cost (see the README's M5 notes).

``GameTree`` walks the tree **once** and stores it as arrays, in
breadth-first order so that every depth level is a contiguous slice and every
parent comes before its children:

* ``parent[n]``, ``action[n]``: how node ``n`` was reached (``action`` is the
  index into the parent's legal actions or chance outcomes)
* ``kind[n]``: terminal, chance, or decision; ``player[n]``, ``infoset[n]``
* ``chance_prob[n]``: probability of the edge into ``n`` if its parent is chance
* ``payoff[n]``: player 0's payoff at terminal nodes (0 elsewhere)

Strategies are (num_infosets, max_actions) matrices with ``action_mask``
marking the legal entries. With that, one pass of CFR or best response is a
loop over depth levels where each step is a handful of vectorized operations,
with no Python work per node.
"""

from __future__ import annotations

import numpy as np

from pokerbot.games.base import Game

TERMINAL, CHANCE_NODE, DECISION = 0, 1, 2
# Who chose the edge into a node: player 0, player 1, or chance.
BY_P0, BY_P1, BY_CHANCE = 0, 1, 2


class GameTree:
    def __init__(self, game: Game):
        self.game = game
        parent, action, chance_prob, depth = [-1], [-1], [1.0], [0]
        kind, player, infoset, payoff = [], [], [], []
        self.infoset_keys: list[str] = []
        self.infoset_actions: list[list] = []
        infoset_player, infoset_depth = [], []
        key_index: dict[str, int] = {}

        states = [game.initial_state()]
        i = 0
        while i < len(states):  # breadth-first: `states` grows as we go
            state = states[i]
            if game.is_terminal(state):
                kind.append(TERMINAL)
                player.append(-1)
                infoset.append(-1)
                payoff.append(game.payoff(state, 0))
                children = []
            elif game.is_chance(state):
                kind.append(CHANCE_NODE)
                player.append(-1)
                infoset.append(-1)
                payoff.append(0.0)
                children = game.chance_outcomes(state)
            else:
                p = game.current_player(state)
                key = game.info_set_key(state)
                actions = game.legal_actions(state)
                if key not in key_index:
                    key_index[key] = len(self.infoset_keys)
                    self.infoset_keys.append(key)
                    self.infoset_actions.append(list(actions))
                    infoset_player.append(p)
                    infoset_depth.append(depth[i])
                elif infoset_depth[key_index[key]] != depth[i]:
                    infoset_depth[key_index[key]] = -1  # spans depths; see best_response
                kind.append(DECISION)
                player.append(p)
                infoset.append(key_index[key])
                payoff.append(0.0)
                children = [(a, 1.0) for a in actions]
            for k, (outcome, prob) in enumerate(children):
                states.append(game.next_state(state, outcome))
                parent.append(i)
                action.append(k)
                chance_prob.append(prob)
                depth.append(depth[i] + 1)
            i += 1

        self.num_nodes = len(states)
        self.parent = np.array(parent)
        self.action = np.array(action)
        self.chance_prob = np.array(chance_prob, dtype=np.float64)
        self.kind = np.array(kind)
        self.player = np.array(player)
        self.infoset = np.array(infoset)
        self.payoff = np.array(payoff, dtype=np.float64)
        depth = np.array(depth)
        self.level_bounds = np.searchsorted(depth, np.arange(depth.max() + 2))

        self.num_infosets = len(self.infoset_keys)
        self.num_actions = np.array([len(a) for a in self.infoset_actions])
        self.max_actions = int(self.num_actions.max())
        self.action_mask = np.arange(self.max_actions) < self.num_actions[:, None]
        self.infoset_player = np.array(infoset_player)
        self.key_index = key_index
        self._same_depth_infosets = min(infoset_depth) >= 0

        # Per-edge data, indexed by child node (the root has no edge).
        par = np.maximum(self.parent, 0)
        self.chooser = np.where(
            self.kind[par] == CHANCE_NODE, BY_CHANCE, np.where(self.player[par] == 0, BY_P0, BY_P1)
        )
        self.chooser[0] = BY_CHANCE
        # Flat (infoset, action) index for edges out of decision nodes.
        self.edge_slot = np.where(
            self.chooser != BY_CHANCE, self.infoset[par] * self.max_actions + self.action, -1
        )

    # --- Strategies ---------------------------------------------------------------

    def levels(self):
        """(parent level slice, child level slice) pairs, top to bottom."""
        b = self.level_bounds
        return [(slice(b[d], b[d + 1]), slice(b[d + 1], b[d + 2])) for d in range(len(b) - 2)]

    def uniform_strategy(self) -> np.ndarray:
        return self.action_mask / self.num_actions[:, None]

    def strategy_matrix(self, strategy: dict[str, list[float]]) -> np.ndarray:
        """Convert {info set key: probabilities} to a matrix; missing keys are uniform."""
        matrix = self.uniform_strategy()
        for key, probs in strategy.items():
            idx = self.key_index.get(key)
            if idx is not None:
                matrix[idx, : len(probs)] = probs
        return matrix

    def strategy_dict(self, matrix: np.ndarray) -> dict[str, list[float]]:
        return {
            key: matrix[i, : self.num_actions[i]].tolist()
            for i, key in enumerate(self.infoset_keys)
        }

    def regret_matching(self, regret: np.ndarray) -> np.ndarray:
        """Row-wise regret matching with a uniform fallback (same rule as M0)."""
        positive = np.maximum(regret, 0) * self.action_mask
        total = positive.sum(axis=1, keepdims=True)
        uniform = self.uniform_strategy()
        return np.divide(positive, total, out=uniform, where=total > 0)

    def normalize(self, sums: np.ndarray) -> np.ndarray:
        """Turn accumulated strategy sums into probabilities (uniform where all zero)."""
        return self.regret_matching(sums)  # sums are non-negative, so this just normalizes

    def edge_probs(self, strategy: np.ndarray) -> np.ndarray:
        """Probability of the edge into each node under `strategy` (chance edges included)."""
        probs = self.chance_prob.copy()
        decision = self.edge_slot >= 0
        probs[decision] = strategy.ravel()[self.edge_slot[decision]]
        return probs

    # --- Passes over the tree ----------------------------------------------------

    def reach(self, edge: np.ndarray) -> np.ndarray:
        """Reach probabilities, shape (3, num_nodes): player 0, player 1, chance."""
        reach = np.ones((3, self.num_nodes))
        for _, children in self.levels():
            par = self.parent[children]
            chooser = self.chooser[children]
            for who in (BY_P0, BY_P1, BY_CHANCE):
                factor = np.where(chooser == who, edge[children], 1.0)
                reach[who, children] = reach[who, par] * factor
        return reach

    def values(self, edge: np.ndarray, leaf: np.ndarray | None = None) -> np.ndarray:
        """Expected value of every node (player 0's payoff by default), bottom up."""
        values = (self.payoff if leaf is None else leaf).copy()
        for parents, children in reversed(self.levels()):
            values[parents] += np.bincount(
                self.parent[children] - parents.start,
                weights=edge[children] * values[children],
                minlength=parents.stop - parents.start,
            )
        return values

    def expected_value(self, strategy: np.ndarray, player: int = 0) -> float:
        value = self.values(self.edge_probs(strategy))[0]
        return value if player == 0 else -value

    def best_response_value(self, strategy: np.ndarray, player: int) -> float:
        """Value for `player` best-responding to the opponent's part of `strategy`.

        Same algorithm as ``eval.exploitability``: at each of the responder's info
        sets, pick the action with the highest opponent-and-chance-reach-weighted
        value, working bottom up. Requires each info set's nodes to share a depth
        (true for every game here) so an info set is decided within one level.
        """
        if not self._same_depth_infosets:
            raise ValueError("info sets span several depths; use eval.exploitability instead")
        edge = self.edge_probs(strategy)
        own = BY_P0 if player == 0 else BY_P1
        # Opponent-and-chance reach: the responder's own edges count as 1.
        weight = np.ones(self.num_nodes)
        for _, children in self.levels():
            factor = np.where(self.chooser[children] == own, 1.0, edge[children])
            weight[children] = weight[self.parent[children]] * factor

        values = self.payoff * (1.0 if player == 0 else -1.0)
        slots = self.num_infosets * self.max_actions
        for parents, children in reversed(self.levels()):
            child = np.arange(children.start, children.stop)
            par = self.parent[child]
            mine = self.chooser[child] == own
            # Chance and opponent nodes: expectation over their edges.
            values[parents] += np.bincount(
                par[~mine] - parents.start,
                weights=edge[child[~mine]] * values[child[~mine]],
                minlength=parents.stop - parents.start,
            )
            if mine.any():
                # Responder nodes: best action per info set, then copy that child's value.
                c, p = child[mine], par[mine]
                score = np.bincount(
                    self.edge_slot[c], weights=weight[p] * values[c], minlength=slots
                ).reshape(self.num_infosets, self.max_actions)
                best = np.where(self.action_mask, score, -np.inf).argmax(axis=1)
                chosen = self.action[c] == best[self.infoset[p]]
                values[p[chosen]] = values[c[chosen]]
        return float(values[0])

    def exploitability(self, strategy: np.ndarray) -> float:
        """(BR value for player 0 + BR value for player 1) / 2, in chips per hand."""
        return (self.best_response_value(strategy, 0) + self.best_response_value(strategy, 1)) / 2
