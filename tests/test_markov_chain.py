import pytest

from stochastics.errors import InvalidTransitionMatrix
from stochastics.models import MarkovChain
from stochastics.models.custom_types import MarkovComponents
from stochastics.models.custom_types import TransitionMatrix


# TODO: Test if creation if invalid matrix works
def test_raises_exception_when_missing_arguments() -> None:
    with pytest.raises(TypeError):
        MarkovChain()


@pytest.mark.parametrize(
    "invalid_trans_matrix", [[[0.1, 0.3], [0.1, 0.9]], [[0.1], [0.1, 0.9]]]
)
def test_raises_exception_when_passing_invalid_matrix(
    invalid_trans_matrix: TransitionMatrix,
) -> None:
    with pytest.raises(InvalidTransitionMatrix):
        MarkovChain(initial_prob=[0.1, 0.2], transition_matrix=invalid_trans_matrix)


@pytest.mark.usefixtures("initial_information")
def test_has_proper_representation(initial_information: MarkovComponents) -> None:
    initial, trans_matrix = initial_information
    expected_repr = {"initial": initial, "transition_matrix": trans_matrix}
    mc = MarkovChain(initial, trans_matrix)
    assert str(expected_repr) == repr(mc)


@pytest.mark.usefixtures("initial_information")
def test_can_calculate_prob_from_sequence(
    initial_information: MarkovComponents,
) -> None:
    ini, trans_matrix = initial_information
    mc = MarkovChain(ini, trans_matrix)

    assert mc.get_probability_from_sequence([0, 1, 0, 2, 0]) == 0.00378


@pytest.mark.usefixtures("initial_information")
def test_can_calculate_prob_with_initial_state(
    initial_information: MarkovComponents,
) -> None:
    ini, trans_matrix = initial_information
    mc = MarkovChain(ini, trans_matrix)

    assert mc.get_probability_from_sequence([1, 0], initial_state=0) == 0.18
