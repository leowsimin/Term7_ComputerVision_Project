import tensorflow as tf
from tensorflow.python.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.python.keras import backend as K


# NOTE: ryan: had to modify this quite a bit to add the l2 regularization - l2_reg is now a parameter, and they are added to the depthwise convolution and standard convolutional layers

class BlazeBlock(tf.keras.Model):
    def __init__(self, block_num=3, channel=48, channel_padding=1, name_prefix="", l2_reg=1e-4):
        super(BlazeBlock, self).__init__()
        self.downsample_a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, strides=(2, 2), padding='same', activation=None,
                depthwise_regularizer=tf.keras.regularizers.L2(l2_reg),
                name=name_prefix + "downsample_a_depthwise"
            ),
            tf.keras.layers.Conv2D(
                filters=channel, kernel_size=1, activation=None,
                kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                name=name_prefix + "downsample_a_conv1x1"
            )
        ], name=name_prefix + "downsample_a")  # Explicit Sequential naming
        if channel_padding:
            self.downsample_b = tf.keras.models.Sequential([
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), name=name_prefix + "downsample_b_maxpool"),
                tf.keras.layers.Conv2D(
                    filters=channel, kernel_size=1, activation=None,
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                    name=name_prefix + "downsample_b_conv1x1"
                )
            ], name=name_prefix + "downsample_b")
        else:
            self.downsample_b = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name=name_prefix + "downsample_b_maxpool")

        self.conv = []
        for i in range(block_num):
            self.conv.append(tf.keras.models.Sequential([
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3, padding='same', activation=None,
                    depthwise_regularizer=tf.keras.regularizers.L2(l2_reg),
                    name=name_prefix + f"conv_block_{i+1}_depthwise"
                ),
                tf.keras.layers.Conv2D(
                    filters=channel, kernel_size=1, activation=None,
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                    name=name_prefix + f"conv_block_{i+1}_conv1x1"
                )
            ], name=name_prefix + f"conv_block_{i+1}"))

    def call(self, x):
        x = tf.keras.activations.relu(self.downsample_a(x) + self.downsample_b(x))
        for conv_block in self.conv:
            x = tf.keras.activations.relu(x + conv_block(x))
        return x


class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio
        self.shared_layer_one = Dense(self.ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        self.shared_layer_two = Dense(1, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    def call(self, input_feature):
        # Get the dynamic shape of the input tensor
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = tf.shape(input_feature)[channel_axis]

        # Global average pooling
        avg_pool = GlobalAveragePooling2D()(input_feature)    
        avg_pool = tf.reshape(avg_pool, (-1, 1, 1, channel)) 
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        # Global max pooling
        max_pool = GlobalMaxPooling2D()(input_feature)
        max_pool = tf.reshape(max_pool, (-1, 1, 1, channel)) 
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        # Combine both avg_pool and max_pool
        cbam_feature = Add()([avg_pool, max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)
        
        # Adjust for 'channels_first' format if needed
        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
        # Apply attention
        return multiply([input_feature, cbam_feature])

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

        # Define Conv2D layer in __init__, not in call
        self.conv = Conv2D(filters=1,
                           kernel_size=self.kernel_size,
                           strides=1,
                           padding='same',
                           activation='sigmoid',
                           kernel_initializer='he_normal',
                           use_bias=False)

    def call(self, input_feature):
        # Get the dynamic shape of the input tensor
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = tf.shape(input_feature)[channel_axis]

        # Adjust input_feature for 'channels_first' format if needed
        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((2, 3, 1))(input_feature)
        else:
            cbam_feature = input_feature

        # Apply average and max pooling
        avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(cbam_feature)
        max_pool = Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(cbam_feature)

        # Concatenate the pooled features
        concat = Concatenate(axis=3)([avg_pool, max_pool])

        # Apply Conv2D to generate attention map
        cbam_feature = self.conv(concat)

        # Adjust for 'channels_first' format if needed
        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)

        # Apply attention
        return multiply([input_feature, cbam_feature])