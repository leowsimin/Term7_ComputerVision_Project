import tensorflow as tf
from layers import BlazeBlock, ChannelAttention, SpatialAttention
from config import num_joints

class BlazePose():
    def __init__(self, l2_reg=1e-4, dropout_rate=0):  # NOTE: modification - added dropout_rate as a parameter, increased l2_reg from 0 to 1e-4
        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=24, kernel_size=3, strides=(2, 2), padding='same', activation='relu',
                kernel_regularizer=tf.keras.regularizers.L2(l2_reg)
            ),
            tf.keras.layers.Dropout(rate=0)  # NOTE: Add dropout - changed to 0 in earlier layers because model can't train well
        ])
         
        # separable convolution (MobileNet)
        self.conv2_1 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation=None, 
                                            depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=24, kernel_size=1, activation=None, 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Dropout(rate=0)  # NOTE: Add dropout
        ])
        self.conv2_2 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation=None, 
                                            depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=24, kernel_size=1, activation=None, 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Dropout(rate=0)  # NOTE: Add dropout
        ])

        #  ---------- Heatmap branch ----------
        self.conv3 = BlazeBlock(block_num=3, channel=48, l2_reg=l2_reg)  # input res: 128
        self.conv4 = BlazeBlock(block_num=4, channel=96, l2_reg=l2_reg)  # input res: 64
        self.conv5 = BlazeBlock(block_num=5, channel=192, l2_reg=l2_reg) # input res: 32
        self.conv6 = BlazeBlock(block_num=6, channel=288, l2_reg=l2_reg) # input res: 16

        self.conv7a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None, 
                                            depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation="relu", 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2DTranspose(filters=48, kernel_size=2, strides=2, padding="same", activation="relu", 
                                             kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Dropout(rate=0)  # NOTE: Add dropout
        ])
        self.conv7b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None, 
                                            depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation="relu", 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Dropout(rate=0)  # NOTE: Add dropout
        ])

        self.conv8a = tf.keras.models.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=48, kernel_size=2, strides=2, padding="same", activation="relu", 
                                             kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Dropout(rate=0)  # NOTE: Add dropout
        ])
        self.conv8b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None, 
                                            depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation="relu", 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Dropout(rate=0)  # NOTE: Add dropout
        ])
        
        self.conv9a = tf.keras.models.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=48, kernel_size=2, strides=2, padding="same", activation="relu", 
                                             kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Dropout(rate=dropout_rate)  # NOTE: Add dropout
        ])
        self.conv9b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None, 
                                            depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation="relu", 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Dropout(rate=dropout_rate)  # NOTE: Add dropout
        ])

        self.conv10a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None, 
                                            depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=8, kernel_size=1, activation="relu", 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=2, strides=2, padding="same", activation="relu", 
                                             kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Dropout(rate=dropout_rate)  # NOTE: Add dropout
        ])
        self.conv10b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None, 
                                            depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=8, kernel_size=1, activation="relu", 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Dropout(rate=dropout_rate)  # NOTE: Add dropout
        ])
        
        self.conv11 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None, 
                                            depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=8, kernel_size=1, activation="relu", 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=num_joints, kernel_size=3, padding="same", activation=None, 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
        ])

        # ---------- Regression branch ----------
        # NOTE: modification - Added dropout to the regression branch

        self.convMODIFICATIONa = BlazeBlock(block_num=3, channel=48, name_prefix="regression_convMODa_", l2_reg=l2_reg)

        self.convMODIFICATIONb = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None, 
                                            depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation="relu", 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Dropout(rate=dropout_rate)  # NOTE: Add dropout
        ])

        self.conv12a = BlazeBlock(block_num=4, channel=96, name_prefix="regression_conv12a_", l2_reg=l2_reg)
        
        self.conv12b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None, 
                                            depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=96, kernel_size=1, activation="relu", 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Dropout(rate=dropout_rate)  # NOTE: Add dropout
        ])

        self.conv13a = BlazeBlock(block_num=5, channel=192, name_prefix="regression_conv13a_", l2_reg=l2_reg)
        
        self.conv13b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None, 
                                            depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=192, kernel_size=1, activation="relu", 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Dropout(rate=dropout_rate)  # NOTE: Add dropout
        ])

        self.conv14a = BlazeBlock(block_num=6, channel=288, name_prefix="regression_conv14a_", l2_reg=l2_reg)
        
        self.conv14b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None, 
                                            depthwise_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Conv2D(filters=288, kernel_size=1, activation="relu", 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Dropout(rate=0)  # NOTE: Add dropout
        ])

        self.conv15 = tf.keras.models.Sequential([
            BlazeBlock(block_num=7, channel=288, channel_padding=0, name_prefix="regression_conv15a_", l2_reg=l2_reg),
            BlazeBlock(block_num=7, channel=288, channel_padding=0, name_prefix="regression_conv15b_", l2_reg=l2_reg)
        ])

        self.conv16 = tf.keras.models.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=2*num_joints, activation=None, 
                                  kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Reshape((num_joints, 2)),
            tf.keras.layers.Dropout(rate=0)  # NOTE: Add dropout
        ])

        self.conv17 = tf.keras.models.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=num_joints, activation="sigmoid", 
                                  kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.Reshape((num_joints, 1)),
            tf.keras.layers.Dropout(rate=0)  # NOTE: Add dropout
        ])
        
        # # Grouping layers - for easy freezing
        # self.heatmap_layers = [
        #     self.conv3, self.conv4, self.conv5, self.conv6, self.conv7a, self.conv7b,
        #     self.conv8a, self.conv8b, self.conv9a, self.conv9b, self.conv10a, self.conv10b,
        #     self.conv11
        # ]
        # self.regression_layers = [
        #     self.convMODIFICATIONa, self.convMODIFICATIONb, self.conv12a, self.conv12b,
        #     self.conv13a, self.conv13b, self.conv14a, self.conv14b, self.conv15,
        #     self.conv16, self.conv17
        # ]

    def call(self):
        input_x = tf.keras.layers.Input(shape=(256, 256, 3))

        # ---------- backbone ----------
        # shape = (1, 256, 256, 3)
        x = self.conv1(input_x)
        # shape x = (1, 128, 128, 24)
        
        x = x + self.conv2_1(x)   # <-- skip connection
        x = tf.keras.activations.relu(x)
        x = x + self.conv2_2(x)
        y0 = tf.keras.activations.relu(x)

        # ---------- heatmap branch ----------
        y1 = self.conv3(y0)  # output res: 64
        y2 = self.conv4(y1)  # output res: 32
        y3 = self.conv5(y2)  # output res: 16
        y4 = self.conv6(y3)  # output res: 8

        x = self.conv7a(y4) + self.conv7b(y3)
        x = self.conv8a(x) + self.conv8b(y2)
        x = self.conv9a(x) + self.conv9b(y1)
        x = self.conv10a(x) + self.conv10b(y0)
        
        heatmap = tf.keras.activations.sigmoid(self.conv11(x))

        # Stop gradient for regression
        x = tf.keras.ops.stop_gradient(x)
        y1 = tf.keras.ops.stop_gradient(y1)
        y2 = tf.keras.ops.stop_gradient(y2)
        y3 = tf.keras.ops.stop_gradient(y3)
        y4 = tf.keras.ops.stop_gradient(y4)

        # ---------- regression branch ----------
        x = self.convMODIFICATIONa(x) + self.convMODIFICATIONb(y1) # NOTE: Added layers
        x = self.conv12a(x) + self.conv12b(y2)
        x = self.conv13a(x) + self.conv13b(y3)
        x = self.conv14a(x) + self.conv14b(y4)
        x = self.conv15(x)

        coordinates = self.conv16(x)
        visibility = self.conv17(x)
        result = [heatmap, coordinates, visibility]

        return tf.keras.Model(inputs=input_x, outputs=result)
