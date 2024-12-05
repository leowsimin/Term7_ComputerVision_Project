import tensorflow as tf
from tensorflow.python.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.python.keras import backend as K


class BlazeBlock(tf.keras.Model):
    def __init__(self, block_num = 3, channel = 48, channel_padding = 1, name_prefix=""):
        super(BlazeBlock, self).__init__()
        # <----- downsample ----->
        self.downsample_a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=(2, 2), padding='same', activation=None, name=name_prefix + "downsample_a_depthwise"),
            tf.keras.layers.Conv2D(filters=channel, kernel_size=1, activation=None, name=name_prefix + "downsample_a_conv1x1")
        ])
        if channel_padding:
            self.downsample_b = tf.keras.models.Sequential([
                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(filters=channel, kernel_size=1, activation=None)
            ])
        else:
            self.downsample_b = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.conv = list()
        for i in range(block_num):
            self.conv.append(tf.keras.models.Sequential([
                tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation=None, name=name_prefix + "conv_block_{}".format(i+1)),
                tf.keras.layers.Conv2D(filters=channel, kernel_size=1, activation=None)
            ]))

    def call(self, x):
        x = tf.keras.activations.relu(self.downsample_a(x) + self.downsample_b(x))
        for i in range(len(self.conv)):
            x = tf.keras.activations.relu(x + self.conv[i](x))
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