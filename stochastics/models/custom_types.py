from typing import Sequence
from typing import Tuple
from typing import TypedDict


# Markov Chain Types
InitialProbabilities = Sequence[float]
TransitionMatrix = Sequence[Sequence[float]]
Transition = Tuple[int, int]
MarkovComponents = Tuple[InitialProbabilities, TransitionMatrix]


class MarkovRepr(TypedDict):
    initial_prob: InitialProbabilities
    transition_matrix: TransitionMatrix
