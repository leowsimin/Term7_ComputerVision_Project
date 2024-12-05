from utils.metrics import *
import config
import pytest

def test_update_state():
    y_true = tf.constant([
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7],
            [8, 8], [9, 9], [10, 10], [11, 11], [12, 12], [13, 13]]
    ], dtype=tf.float32)

    head_length = Eclidian2(y_true[:, 12], y_true[:, 13])

    # first image will hv 14 correct
    y_pred = y_true + tf.random.uniform(y_true.shape, minval=-head_length*0.2, maxval=head_length*0.2)

    # Run the update_state function
    obj = PCKMetric()
    obj.update_state(y_true, y_pred)
    res = obj.result()

    # Check that correct_keypoints and total_keypoints have expected values
    assert res == 1

def test_update_state_2():
    y_true = tf.constant([
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7],
            [8, 8], [9, 9], [10, 10], [11, 11], [12, 12], [13, 13]]
    ], dtype=tf.float32)

    head_length = Eclidian2(y_true[:, 12], y_true[:, 13])

    # first image will hv 0 correct
    y_pred = y_true + tf.random.uniform(y_true.shape, minval=head_length*0.7, maxval=head_length*1)

    # Run the update_state function
    obj = PCKMetric()
    obj.update_state(y_true, y_pred)
    res = obj.result()

    # Check that correct_keypoints and total_keypoints have expected values
    assert res == 0