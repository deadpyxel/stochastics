import pytest

from stochastics.errors import InvalidTransitionMatrix
from stochastics.models import MarkovChain
from stochastics.models.custom_types import MarkovComponents
from stochastics.models.custom_types import TransitionMatrix


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


@pytest.mark.usefixtures("initial_information")
@pytest.mark.parametrize(
    ("n", "expectation"),
    [
        (2, [[0.01, 0.04, 0.49], [0.81, 0.01, 0.0], [0.01, 0.64, 0.01]]),
        (3, [[0.001, 0.008, 0.343], [0.729, 0.001, 0.0], [0.001, 0.512, 0.001]]),
        (
            5,
            [[1e-05, 0.00032, 0.16807], [0.59049, 1e-05, 0.0], [1e-05, 0.32768, 1e-05]],
        ),
        (100, [[0.0, 0.0, 0.0], [3e-05, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    ],
)
def test_can_calculate_transition_matrix_n_steps(
    initial_information: MarkovComponents, n: int, expectation: TransitionMatrix
) -> None:
    ini, trans_matrix = initial_information
    mc = MarkovChain(ini, trans_matrix)

    p_matrix = mc.get_n_steps_probability(n)
    print(p_matrix)
    assert p_matrix == expectation
