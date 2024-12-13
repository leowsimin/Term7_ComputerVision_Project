import tensorflow as tf

def loss_func_bce_negative_joint(y_true, y_pred):
    # y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    epsilon = tf.keras.backend.epsilon()
    term_0 = (1 - y_true) * tf.math.log(y_pred + epsilon)  # Cancels out when target is 1 
    term_1 = y_true * tf.math.log(1 - y_pred + epsilon) # Cancels out when target is 0
    bce = -(term_0 + term_1)
    loss_only_at_joint = tf.where(tf.greater(y_true, 0), bce * 1, bce * 0)
    return 10 * (tf.reduce_mean(loss_only_at_joint))