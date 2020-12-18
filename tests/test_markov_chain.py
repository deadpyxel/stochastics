import pytest

from stochastics.errors import InvalidTransitionMatrix
from stochastics.models import MarkovChain
from stochastics.models.custom_types import MarkovComponents
from stochastics.models.custom_types import MarkovRepr
from stochastics.models.custom_types import TransitionMatrix


@pytest.mark.parametrize(
    "invalid_trans_matrix",
    [
        pytest.param([[0.1, 0.3], [0.1, 0.9]], id="doesnt_sum_to_1"),
        pytest.param([[0.1], [0.1, 0.9]], id="missing_value"),
    ],
)
def test_raises_exception_when_passing_invalid_matrix(
    invalid_trans_matrix: TransitionMatrix,
) -> None:
    with pytest.raises(InvalidTransitionMatrix):
        MarkovChain(initial_prob=[0.1, 0.2], transition_matrix=invalid_trans_matrix)


def test_can_create_markovchain_from_dict() -> None:
    data_dict: MarkovRepr = {
        "initial_prob": [0.1, 0.2],
        "transition_matrix": [
            [0.1, 0.9],
            [0.8, 0.2],
        ],
    }
    MarkovChain.from_dict(data_dict)


def test_can_create_markovchain_from_str() -> None:
    input_str = "0.1 0.2\n0.1 0.9\n0.8 0.2"
    MarkovChain.from_str(input_str)


@pytest.mark.usefixtures("initial_information")
def test_has_proper_representation(initial_information: MarkovComponents) -> None:
    initial, trans_matrix = initial_information
    expected_repr = {"initial": initial, "transition_matrix": trans_matrix}
    mc = MarkovChain(initial, trans_matrix)
    assert str(expected_repr) == repr(mc), "An incorrect representation was found"


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
        pytest.param(
            2, [[0.26, 0.6, 0.14], [0.18, 0.19, 0.63], [0.74, 0.18, 0.08]], id="p2"
        ),
        pytest.param(
            3,
            [[0.58, 0.224, 0.196], [0.252, 0.559, 0.189], [0.244, 0.23, 0.526]],
            id="p3",
        ),
        pytest.param(
            5,
            [
                [0.33616, 0.42584, 0.238],
                [0.306, 0.29143, 0.40257],
                [0.49408, 0.28478, 0.22114],
            ],
            id="p5",
        ),
        pytest.param(
            100,
            [
                [0.37156, 0.33944, 0.28901],
                [0.37157, 0.33943, 0.289],
                [0.37156, 0.33944, 0.28899],
            ],
            id="p100",
        ),
    ],
)
def test_can_calculate_transition_matrix_n_steps(
    initial_information: MarkovComponents, n: int, expectation: TransitionMatrix
) -> None:
    ini, trans_matrix = initial_information
    mc = MarkovChain(ini, trans_matrix)

    p_matrix = mc.get_n_steps_probability_matrix(n)
    assert p_matrix == expectation, "The expected result was not found."


@pytest.mark.usefixtures("initial_information")
def test_p0_matrix_is_identity(initial_information: MarkovComponents) -> None:
    ini, trans_matrix = initial_information
    mc = MarkovChain(ini, trans_matrix)

    p0_matrix = mc.get_n_steps_probability_matrix(n=0)
    expected_result = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert (
        p0_matrix == expected_result
    ), "non-identity matrix was returned for p^0 operation"


@pytest.mark.usefixtures("initial_information")
def test_can_get_inconditional_proba_array(
    initial_information: MarkovComponents,
) -> None:
    ini, trans_matrix = initial_information
    mc = MarkovChain(ini, trans_matrix)

    pi2 = mc.calc_inconditional_probability(n=2)
    expected_vector = [0.372, 0.31, 0.318]
    assert pi2 == expected_vector, "Wrong probability vector"
