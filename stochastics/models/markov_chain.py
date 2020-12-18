from __future__ import annotations

from typing import List
from typing import Optional
from typing import Sequence

from .custom_types import InitialProbabilities
from .custom_types import MarkovRepr
from .custom_types import Transition
from .custom_types import TransitionMatrix
from stochastics.errors import InvalidTransitionMatrix
from stochastics.utils import validate_transition_matrix


class MarkovChain:
    def __init__(
        self, initial_prob: InitialProbabilities, transition_matrix: TransitionMatrix
    ) -> None:
        self._initial = initial_prob
        if validate_transition_matrix(transition_matrix, n_states=len(initial_prob)):
            self._transition_matrix = transition_matrix
        else:
            raise InvalidTransitionMatrix(
                "The transition matrix is invalid, please check and run again"
            )
        self._n_states = len(self._initial)

    @classmethod
    def from_dict(cls, data_dict: MarkovRepr) -> MarkovChain:
        return cls(**data_dict)

    @classmethod
    def from_str(cls, input_data: str) -> MarkovChain:
        values = [
            [float(val) for val in line.split()] for line in input_data.split("\n")
        ]
        initial_prob, *transition_matrix = values
        print(initial_prob, transition_matrix)
        return cls(initial_prob, transition_matrix)

    def _get_probability_step(self, origin_state: int, target_state: int) -> float:
        return self._transition_matrix[origin_state][target_state]

    def get_probability_from_sequence(
        self,
        state_sequence: List[int],
        initial_state: Optional[int] = None,
        precision: int = 5,
    ) -> float:
        p = 1.0
        if initial_state is None:
            curr_state = state_sequence.pop(0)
            p = self._initial[curr_state]
        else:
            curr_state = initial_state
        for state in state_sequence:
            p *= self._get_probability_step(curr_state, state)
            curr_state = state
        return round(p, precision)

    def __repr__(self) -> str:
        return str(
            {"initial": self._initial, "transition_matrix": self._transition_matrix}
        )

    def get_n_steps_probability_matrix(
        self, n: int, precision: int = 5
    ) -> TransitionMatrix:
        if n == 0:
            return [
                [1.0 if i == j else 0.0 for i in range(self._n_states)]
                for j in range(self._n_states)
            ]
        p_matrix = self._transition_matrix
        for _ in range(n - 1):
            p_matrix = [
                [
                    round(sum(a * b for a, b in zip(line, col)), precision)
                    for col in zip(*p_matrix)
                ]
                for line in self._transition_matrix
            ]
        return p_matrix

    def get_n_steps_probability(
        self, n: int, transition: Transition, precision: int = 5
    ) -> float:
        prob_matrix = self.get_n_steps_probability_matrix(n - 1, precision)
        og_state, dest_state = transition
        p = 0.0
        for i in range(self._n_states):
            if i in transition:
                continue
            p += self._get_probability_step(og_state, i) * prob_matrix[i][dest_state]
        return p

    def calc_inconditional_probability(
        self, n: int, precision: int = 5
    ) -> Sequence[float]:
        pn_matrix = self.get_n_steps_probability_matrix(n, precision)
        inconditional_matrix = [
            round(sum(p * el for p, el in zip(self._initial, col)), precision)
            for col in zip(*pn_matrix)
        ]
        return inconditional_matrix

    def get_prob_inconditional(self, state: int, n: int) -> float:
        pn_matrix = self.get_n_steps_probability_matrix(n)
        state_prob = [line[state] for line in pn_matrix]
        return sum(p * prob for p, prob in zip(self._initial, state_prob))
