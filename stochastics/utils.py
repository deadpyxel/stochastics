from .models.custom_types import TransitionMatrix


def validate_transition_matrix(trans_matrix: TransitionMatrix, n_states: int) -> bool:
    """Validates a transition matrix by the probability theory criteria

    Args:
        trans_matrix (TransitionMatrix): the transition amtrix for validation
        n_states (int): number of states expected in the transition amtrix

    Returns:
        bool: True if all lines in the transition matrix sum 1.0 and its a square matrix
    """
    return (
        all(sum(line) == 1.0 and len(line) == n_states for line in trans_matrix)
        and len(trans_matrix) == n_states
    )
