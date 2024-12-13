import tensorflow as tf
from layers import (
    BlazeBlock,
    # PatchEmbedding,
    # PositionalEmbedding,
    # TransformerBlock,
    CBAM,
)
from config import num_joints
from tensorflow.python.keras import backend as K


class CBAM_BlazePose:
    def __init__(self, l2_reg=0):
        self.conv1 = tf.keras.layers.Conv2D(
            filters=24,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
        )

        # separable convolution (MobileNet)
        self.conv2_1 = tf.keras.models.Sequential(
            [
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3,
                    padding="same",
                    activation=None,
                    depthwise_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
                tf.keras.layers.Conv2D(
                    filters=24,
                    kernel_size=1,
                    activation=None,
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
            ]
        )
        self.conv2_2 = tf.keras.models.Sequential(
            [
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3,
                    padding="same",
                    activation=None,
                    depthwise_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
                tf.keras.layers.Conv2D(
                    filters=24,
                    kernel_size=1,
                    activation=None,
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
            ]
        )

        #  ---------- Heatmap branch ----------
        self.conv3 = BlazeBlock(block_num=3, channel=48)  # input res: 128
        # TODO: Add Transformer Block
        self.cbam3 = CBAM()  # Add CBAM after conv3
        self.conv4 = BlazeBlock(block_num=4, channel=96)  # input res: 64
        self.conv5 = BlazeBlock(block_num=5, channel=192)  # input res: 32
        self.cbam5 = CBAM()  # Add CBAM after conv5
        self.conv6 = BlazeBlock(block_num=6, channel=288)  # input res: 16

        self.conv7a = tf.keras.models.Sequential(
            [
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3,
                    padding="same",
                    activation=None,
                    depthwise_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
                tf.keras.layers.Conv2D(
                    filters=48,
                    kernel_size=1,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
                tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear"),
            ]
        )
        self.conv7b = tf.keras.models.Sequential(
            [
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3,
                    padding="same",
                    activation=None,
                    depthwise_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
                tf.keras.layers.Conv2D(
                    filters=48,
                    kernel_size=1,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
            ]
        )

        self.conv8a = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear"
        )
        self.conv8b = tf.keras.models.Sequential(
            [
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3,
                    padding="same",
                    activation=None,
                    depthwise_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
                tf.keras.layers.Conv2D(
                    filters=48,
                    kernel_size=1,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
            ]
        )

        self.conv9a = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear"
        )
        self.conv9b = tf.keras.models.Sequential(
            [
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3,
                    padding="same",
                    activation=None,
                    depthwise_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
                tf.keras.layers.Conv2D(
                    filters=48,
                    kernel_size=1,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
            ]
        )

        self.conv10a = tf.keras.models.Sequential(
            [
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3,
                    padding="same",
                    activation=None,
                    depthwise_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
                tf.keras.layers.Conv2D(
                    filters=8,
                    kernel_size=1,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
                tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear"),
            ]
        )
        self.conv10b = tf.keras.models.Sequential(
            [
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3,
                    padding="same",
                    activation=None,
                    depthwise_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
                tf.keras.layers.Conv2D(
                    filters=8,
                    kernel_size=1,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
            ]
        )

        # the output layer for heatmap and offset
        self.conv11 = tf.keras.models.Sequential(
            [
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3,
                    padding="same",
                    activation=None,
                    depthwise_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
                tf.keras.layers.Conv2D(
                    filters=8,
                    kernel_size=1,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
                # heatmap
                tf.keras.layers.Conv2D(
                    filters=num_joints,
                    kernel_size=3,
                    padding="same",
                    activation=None,
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
            ]
        )
        self.cbam11 = CBAM()  # Add CBAM after conv11b

        # ---------- Regression branch ----------
        #  shape = (1, 64, 64, 48)
        self.conv12a = BlazeBlock(
            block_num=4, channel=96, name_prefix="regression_conv12a_"
        )  # input res: 64
        self.conv12b = tf.keras.models.Sequential(
            [
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3,
                    padding="same",
                    activation=None,
                    name="regression_conv12b_depthwise",
                    depthwise_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
                tf.keras.layers.Conv2D(
                    filters=96,
                    kernel_size=1,
                    activation="relu",
                    name="regression_conv12b_conv1x1",
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
            ],
            name="regression_conv12b",
        )

        self.conv13a = BlazeBlock(
            block_num=5, channel=192, name_prefix="regression_conv13a_"
        )  # input res: 32
        self.conv13b = tf.keras.models.Sequential(
            [
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3,
                    padding="same",
                    activation=None,
                    name="regression_conv13b_depthwise",
                    depthwise_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
                tf.keras.layers.Conv2D(
                    filters=192,
                    kernel_size=1,
                    activation="relu",
                    name="regression_conv13b_conv1x1",
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
            ],
            name="regression_conv13b",
        )

        self.conv14a = BlazeBlock(
            block_num=6, channel=288, name_prefix="regression_conv14a_"
        )  # input res: 16
        self.conv14b = tf.keras.models.Sequential(
            [
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3,
                    padding="same",
                    activation=None,
                    name="regression_conv14b_depthwise",
                    depthwise_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
                tf.keras.layers.Conv2D(
                    filters=288,
                    kernel_size=1,
                    activation="relu",
                    name="regression_conv14b_conv1x1",
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
            ],
            name="regression_conv14b",
        )

        self.conv15 = tf.keras.models.Sequential(
            [
                BlazeBlock(
                    block_num=7,
                    channel=288,
                    channel_padding=0,
                    name_prefix="regression_conv15a_",
                ),
                BlazeBlock(
                    block_num=7,
                    channel=288,
                    channel_padding=0,
                    name_prefix="regression_conv15b_",
                ),
            ],
            name="regression_conv15",
        )

        # using regression + sigmoid
        self.conv16 = tf.keras.models.Sequential(
            [
                tf.keras.layers.GlobalAveragePooling2D(),
                # shape = (1, 1, 1, 288)
                # coordinates
                tf.keras.layers.Dense(
                    units=2 * num_joints,
                    activation=None,
                    name="regression_final_dense_1",
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
                tf.keras.layers.Reshape((num_joints, 2)),
            ],
            name="coordinates",
        )

        self.conv17 = tf.keras.models.Sequential(
            [
                tf.keras.layers.GlobalAveragePooling2D(),
                # shape = (1, 1, 1, 288)
                # visibility
                tf.keras.layers.Dense(
                    units=num_joints,
                    activation="sigmoid",
                    name="regression_final_dense_2",
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                ),
                tf.keras.layers.Reshape((num_joints, 1)),
            ],
            name="visibility",
        )

    def call(self):
        input_x = tf.keras.layers.Input(shape=(256, 256, 3))  # Keras symbolic tensor

        # shape = (1, 256, 256, 3)
        x = self.conv1(input_x)
        print(f"X after conv1 --> {x.shape}")
        # shape = (1, 128, 128, 24)
        x = x + self.conv2_1(x)  # <-- skip connection
        # x = tf.keras.activations.relu(x)
        print(f"X after conv_2_1 --> {x.shape}")
        x = x + self.conv2_2(x)
        print(f"X after conv_2_2 --> {x.shape}")
        # y0 = tf.keras.activations.relu(x)

        # ---------- ViT Transformer ----------
        # embedded_patches = self.patch_embedding_layer(x)
        # print("Shape of embedded patches:", embedded_patches.shape)

        # embedded_patches_with_position = self.positional_embedding_layer(
        #     embedded_patches
        # )
        # print(
        #     "Shape of embedded patches with positional embeddings:",
        #     embedded_patches_with_position.shape,
        # )

        # transformer_output = embedded_patches_with_position
        # for _ in range(self.num_transformer_layers):  # Stack multiple layers
        #     transformer_output = self.transformer_block(transformer_output)
        # print("Shape after stacking Transformer Blocks:", transformer_output.shape)

        # ## FIX THIS PORTION.
        # # Reshaping the patches into a 2D grid (e.g., 8x8 grid for 64 patches)
        # # transformer_output: Keras Symbolic Tensor

        # # print("transformer_output type:", type(transformer_output))
        # reshaped_patches = self.reshape_layer(transformer_output)  # (None, 8, 8, 128)
        # reshaped_patches = reshaped_patches[:, 0, :, :, :]
        # print("Reshaped size:", reshaped_patches.shape)

        # upsampled_patches = self.conv_trans_upsampling(reshaped_patches)
        # print("Upsampled Shape:", upsampled_patches)

        # output = self.conv1x1(upsampled_patches)
        # print("Output shape:", output.shape)  # (None, 128, 128, 24)

        # ---------- heatmap branch ----------
        # shape = (1, 128, 128, 24)
        y0 = x
        y1 = self.conv3(y0)  # output res: 64
        y1 = self.cbam3(y1)  # CBAM after conv3
        print(f"Y1 shape --> {y1.shape}")
        y2 = self.conv4(y1)  # output res:  32
        y3 = self.conv5(y2)  # output res:  16
        y3 = self.cbam5(y3)  # CBAM after conv5
        y4 = self.conv6(y3)  # output res:  8
        # shape = (1, 8, 8, 288)
        print(
            f"Y2 shape --> {y2.shape}\nY3 shape --> {y3.shape}\nY4 shape --> {y4.shape}"
        )

        x = self.conv7a(y4) + self.conv7b(y3)
        print(f"X conv7a conv7b --> {x.shape}")
        x = self.conv8a(x) + self.conv8b(y2)
        print(f"X conv8a conv8b --> {x.shape}")
        # shape = (1, 32, 32, 96)
        x = self.conv9a(x) + self.conv9b(y1)
        print(f"X conv9a conv9b --> {x.shape}")
        # shape = (1, 64, 64, 48)
        y = self.conv10a(x) + self.conv10b(y0)  # x
        print("After heatmap:", y.shape)
        # y = self.cbam10(y)
        # shape = (1, 128, 128, 8)
        y_intermediate = self.conv11(y)
        ycbam = self.cbam11(y_intermediate)

        heatmap = tf.keras.activations.sigmoid(y_intermediate)
        print(f"Heatmap layer shape: {heatmap.shape}")
        # Stop gradient for regression
        x = tf.keras.ops.stop_gradient(x)
        y2 = tf.keras.ops.stop_gradient(y2)
        print("y2 Shape:", y2.shape)
        y3 = tf.keras.ops.stop_gradient(y3)
        y4 = tf.keras.ops.stop_gradient(y4)

        # ---------- regression branch ----------
        print("Before First Regression Branch:", x.shape)
        x = self.conv12a(x) + self.conv12b(y2)
        print("First Regression Branch:", x.shape)
        # shape = (1, 32, 32, 96)
        x = self.conv13a(x) + self.conv13b(y3)
        print("Second Regression Branch:", x.shape)
        # shape = (1, 16, 16, 192)
        x = self.conv14a(x) + self.conv14b(y4)
        print("Third Regression Branch:", x.shape)
        # shape = (1, 8, 8, 288)
        x = self.conv15(x)
        print("Forth Regression Branch:", x.shape)
        # shape = (1, 2, 2, 288)

        # using linear + sigmoid
        coordinates = self.conv16(x)
        print("Coordinates:", coordinates.shape)
        visibility = self.conv17(x)
        print("Visibility:", visibility.shape)
        result = [heatmap, coordinates, visibility]

        return tf.keras.Model(inputs=input_x, outputs=result)
