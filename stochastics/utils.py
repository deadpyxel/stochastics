from .models.custom_types import TransitionMatrix


def validate_transition_matrix(trans_matrix: TransitionMatrix) -> bool:
    return all(
        sum(line) == 1.0 and len(line) == len(trans_matrix) for line in trans_matrix
    )
