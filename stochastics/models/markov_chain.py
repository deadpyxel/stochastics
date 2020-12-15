from typing import List
from typing import Optional

from .custom_types import InitialProbabilities
from .custom_types import TransitionMatrix
from stochastics.errors import InvalidTransitionMatrix
from stochastics.utils import validate_transition_matrix


class MarkovChain:
    def __init__(
        self, initial_prob: InitialProbabilities, transition_matrix: TransitionMatrix
    ) -> None:
        self._initial = initial_prob
        if validate_transition_matrix(transition_matrix):
            self._transition_matrix = transition_matrix
        else:
            raise InvalidTransitionMatrix(
                "The transition matrix is invalid, please check and run again"
            )

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
