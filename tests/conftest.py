import pytest

from stochastics.models.custom_types import MarkovComponents


@pytest.fixture()
def initial_information() -> MarkovComponents:
    """Initial Information for a MarkovChain

    The initial probabilities for each state and the transition matrix

    Returns:
        MarkovComponents: Components for a MarkovChain object
    """
    ini = [0.3, 0.4, 0.3]
    trans_matrix = [[0.1, 0.2, 0.7], [0.9, 0.1, 0.0], [0.1, 0.8, 0.1]]

    return ini, trans_matrix
