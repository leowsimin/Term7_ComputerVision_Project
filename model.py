import tensorflow as tf
from layers import BlazeBlock, ChannelAttention, SpatialAttention
from config import num_joints

class BlazePose():
    def __init__(self, l2_reg=0):
        self.conv1 = tf.keras.layers.Conv2D(
            filters=24, kernel_size=3, strides=(2, 2), padding='same', activation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(l2_reg)
        )
        self.norm1 = tf.keras.layers.BatchNormalization()
         
        # separable convolution (MobileNet)
        self.conv2_1 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation=None,   depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=24, kernel_size=1, activation=None,   kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
        ])
        self.conv2_2 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation=None,   depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=24, kernel_size=1, activation=None,   kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
        ])

        #  ---------- Heatmap branch ----------
        self.conv3 = BlazeBlock(block_num = 3, channel = 48)    # input res: 128
        self.conv4 = BlazeBlock(block_num = 4, channel = 96)    # input res: 64
        self.conv5 = BlazeBlock(block_num = 5, channel = 192)   # input res: 32
        self.conv6 = BlazeBlock(block_num = 6, channel = 288)   # input res: 16

        self.conv7a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None,   depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation="relu",   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        ])
        self.conv7b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None,   depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation="relu",   kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
        ])

        self.conv8a = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        self.conv8b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None,   depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation="relu",   kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
        ])

        self.conv9a = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        self.conv9b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None,   depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation="relu",   kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
        ])

        self.conv10a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None,   depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=8, kernel_size=1, activation="relu",   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear"
            )
        ])
        self.conv10b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None,   depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=8, kernel_size=1, activation="relu",   kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
        ])

        # the output layer for heatmap and offset
        self.conv11 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None,   depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=8, kernel_size=1, activation="relu",   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            # heatmap
            tf.keras.layers.Conv2D(filters=num_joints, kernel_size=3, padding="same", activation=None,   kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
        ])        
        
        self.conv11b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None,   depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=8, kernel_size=1, activation="relu",   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            # heatmap
            tf.keras.layers.Conv2D(filters=num_joints, kernel_size=3, padding="same", activation=None,   kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
        ])

        # ---------- Regression branch ----------
        #  shape = (1, 64, 64, 48)
        self.conv12a = BlazeBlock(block_num = 4, channel = 96, name_prefix="regression_conv12a_")    # input res: 64
        self.conv12b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None, name="regression_conv12b_depthwise",   depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=96, kernel_size=1, activation="relu", name="regression_conv12b_conv1x1",   kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
        ], name="regression_conv12b")

        self.conv13a = BlazeBlock(block_num = 5, channel = 192, name_prefix="regression_conv13a_")   # input res: 32
        self.conv13b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None, name="regression_conv13b_depthwise",   depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=192, kernel_size=1, activation="relu", name="regression_conv13b_conv1x1",   kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
        ], name="regression_conv13b")

        self.conv14a = BlazeBlock(block_num = 6, channel = 288, name_prefix="regression_conv14a_")   # input res: 16
        self.conv14b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None, name="regression_conv14b_depthwise",   depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=288, kernel_size=1, activation="relu", name="regression_conv14b_conv1x1",   kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
        ], name="regression_conv14b")

        self.conv15 = tf.keras.models.Sequential([
            BlazeBlock(block_num = 7, channel = 288, channel_padding = 0, name_prefix="regression_conv15a_"),
            BlazeBlock(block_num = 7, channel = 288, channel_padding = 0, name_prefix="regression_conv15b_")
        ], name="regression_conv15")

        # using regression + sigmoid
        self.conv16 = tf.keras.models.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            # shape = (1, 1, 1, 288)
            # coordinates
            tf.keras.layers.Dense(units=2*num_joints, activation=None, name="regression_final_dense_1",   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Reshape((num_joints, 2))
        ], name="coordinates")

        self.conv17 = tf.keras.models.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            # shape = (1, 1, 1, 288)
            # visibility
            tf.keras.layers.Dense(units=num_joints, activation="sigmoid", name="regression_final_dense_2",   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Reshape((num_joints, 1))
        ], name="visibility")

    def call(self):
        input_x = tf.keras.layers.Input(shape=(256, 256, 3))

        # x = self.norm1(input_x)

        # shape = (1, 256, 256, 3)
        x = self.conv1(input_x)

        # shape = (1, 128, 128, 24)
        x = x + self.conv2_1(x)   # <-- skip connection
        x = tf.keras.activations.relu(x)
        x = x + self.conv2_2(x)
        y0 = tf.keras.activations.relu(x)

        # ---------- heatmap branch ----------
        # shape = (1, 128, 128, 24)
        y1 = self.conv3(y0) # output res: 64
        y2 = self.conv4(y1) # output res:  32
        y3 = self.conv5(y2) # output res:  16
        y4 = self.conv6(y3) # output res:  8
        # shape = (1, 8, 8, 288)

        x = self.conv7a(y4) + self.conv7b(y3)
        x = self.conv8a(x) + self.conv8b(y2)
        # shape = (1, 32, 32, 96)
        x = self.conv9a(x) + self.conv9b(y1)
        # shape = (1, 64, 64, 48)
        y = self.conv10a(x) + self.conv10b(y0)
        # shape = (1, 128, 128, 8)
        y = self.conv11(y)
        heatmap = tf.keras.activations.sigmoid(y)
        negative_heatmap = tf.keras.activations.sigmoid(y)

        # Stop gradient for regression
        x = tf.keras.ops.stop_gradient(x)
        y2 = tf.keras.ops.stop_gradient(y2)
        y3 = tf.keras.ops.stop_gradient(y3)
        y4 = tf.keras.ops.stop_gradient(y4)

        # ---------- regression branch ----------
        x = self.conv12a(x) + self.conv12b(y2)
        # shape = (1, 32, 32, 96)
        x = self.conv13a(x) + self.conv13b(y3)
        # shape = (1, 16, 16, 192)
        x = self.conv14a(x) + self.conv14b(y4)
        # shape = (1, 8, 8, 288)
        x = self.conv15(x)
        # shape = (1, 2, 2, 288)

        # using linear + sigmoid
        coordinates = self.conv16(x)
        visibility = self.conv17(x)
        result = [heatmap, coordinates, visibility, negative_heatmap]

        return tf.keras.Model(inputs=input_x, outputs=result)