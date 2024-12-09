import functools
import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops, gen_math_ops
import config


def Eclidian2(a, b):
    # Calculate the square of Eclidian distance
    tf.debugging.assert_equal(tf.shape(a), tf.shape(b))
    squared_distance = tf.reduce_sum(tf.square(a - b), axis=-1)
    return squared_distance


class PCKMetric(tf.keras.metrics.Metric):
    def __init__(self, name="pck", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct_keypoints = self.add_weight(
            shape=(), initializer="zeros", name="correct_keypoints"
        )
        self.total_keypoints = self.add_weight(
            shape=(), initializer="zeros", name="total_keypoints"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # print(y_true.shape)
        # print(y_pred.shape)
        y_true = math_ops.cast(y_true[:, :, :2], tf.float32)  # Ground truth keypoints
        y_pred = math_ops.cast(y_pred, tf.float32)  # Predicted keypoints
        head_length = Eclidian2(y_true[:, 12], y_true[:, 13])  # Shape: (batch_size,)
        distances = Eclidian2(y_pred, y_true)  # Shape: (batch_size, 14)

        correct = distances <= (config.pck_metric * head_length[:, tf.newaxis])
        correct = math_ops.cast(correct, self.dtype)

        # Update the total counts
        self.correct_keypoints.assign_add(math_ops.reduce_sum(correct))
        total = tf.cast(tf.math.reduce_prod(tf.shape(correct)), tf.float32)
        self.total_keypoints.assign_add(total)

    def result(self):
        return gen_math_ops.div_no_nan(
            self.correct_keypoints, self.total_keypoints
        )  # Return as percentage


def masked_mse_loss(y_true, y_pred, visibility):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    visibility = tf.cast(visibility, tf.float32)

    # Compute squared error
    squared_error = tf.square(y_true - y_pred)  # Shape: (batch_size, num_joints, 2)
    masked_error = squared_error * visibility  # Mask out occluded joints

    # Mean over all visible joints
    return tf.reduce_mean(masked_error)


def masked_bce_loss(y_true, y_pred, visibility):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    visibility = tf.cast(visibility, tf.float32)

    # Reshape visibility to match heatmap dimensions (batch_size, 1, 1, num_joints)
    visibility = tf.reshape(
        visibility, [tf.shape(visibility)[0], 1, 1, tf.shape(visibility)[1]]
    )
    visibility = tf.broadcast_to(visibility, tf.shape(y_true))  # Match y_true shape

    # Mask y_true and y_pred for occluded joints
    y_true_masked = y_true * visibility
    y_pred_masked = y_pred * visibility

    # Compute BCE loss on masked values (ignores occluded joints)
    bce_loss = tf.keras.backend.binary_crossentropy(y_true_masked, y_pred_masked)

    total_visible = tf.reduce_sum(visibility)  # Total count of visible joints
    return tf.reduce_sum(bce_loss) / (
        total_visible + 1e-7
    )  # normalize by total visible
