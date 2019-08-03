import numpy as np

from printing import target, normalize, normalize_transformation

def test_normalize_transformation():
    actual = normalize_transformation(np.array([-1, 0, 0.5, 1]))
    expected = np.array([0, 0.5, 0.75, 1])
    
    assert np.allclose(actual, expected)

def assert_target_is_correct(initial_rotation, rotation):
    normed_initial_rotation = initial_rotation / 7
    normed_rotation = normalize(rotation % 8, range(8))

    tf = normalize_transformation(np.array([0, normed_rotation, 0, 0]))

    p = [1, 0, 0, 0, 0, 0, 0.5, 0, normed_initial_rotation, 0, 0]
    total_normed_rotation = normed_initial_rotation + normed_rotation 
    if total_normed_rotation > 1:
        total_normed_rotation -= 1 + 1 / 7
    actual = target(np.concatenate([p, tf]))[:11]
    expected = np.array([1, 0, 0, 0, 0, 0, 0.5, 0, total_normed_rotation, 0, 0])

    assert np.allclose(total_normed_rotation, ((initial_rotation + rotation) % 8) / 7)
    assert np.allclose(actual, expected)

def test_rotational_transformation_from_0_to_1():
    initial_rotation = 0
    rotation = 1

    assert_target_is_correct(initial_rotation, rotation)


def test_rotational_transformation_from_0_to_minus_1():
    initial_rotation = 0
    rotation = -1

    assert_target_is_correct(initial_rotation, rotation)


def test_rotational_transformation_from_7_to_0():
    initial_rotation = 7
    rotation = 1

    assert_target_is_correct(initial_rotation, rotation)


def test_rotational_transformation_from_7_to_1():
    initial_rotation = 7
    rotation = 2

    assert_target_is_correct(initial_rotation, rotation)


def test_rotational_transformation_from_1_to_7():
    initial_rotation = 1
    rotation = -2

    assert_target_is_correct(initial_rotation, rotation)

def test_rotational_transformation_from_1_to_6():
    initial_rotation = 1
    rotation = -3

    assert_target_is_correct(initial_rotation, rotation)
