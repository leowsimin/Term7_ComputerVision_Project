import tensorflow as tf
from tensorflow.python.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.python.keras import backend as K


class BlazeBlock(tf.keras.Model):
    def __init__(self, block_num = 3, channel = 48, channel_padding = 1, name_prefix="",l2_reg=0):
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

class CBAM(tf.keras.layers.Layer): 
    def __init__(self, reduction_ratio=16): 
        super(CBAM, self).__init__() 
        self.reduction_ratio = reduction_ratio 
 
    def build(self, input_shape): 
        # Channel attention 
        self.global_avg_pool = GlobalAveragePooling2D() 
        self.global_max_pool = GlobalMaxPooling2D() 
        self.shared_dense1 = Dense( 
            input_shape[-1] // self.reduction_ratio, activation="relu" 
        ) 
        self.shared_dense2 = Dense(input_shape[-1], activation="sigmoid") 
 
        # Spatial attention 
        self.conv_spatial = Conv2D( 
            1, kernel_size=7, strides=1, padding="same", activation="sigmoid" 
        ) 
 
    def call(self, inputs): 
        # Channel attention 
        avg_out = self.shared_dense2(self.shared_dense1(self.global_avg_pool(inputs))) 
        max_out = self.shared_dense2(self.shared_dense1(self.global_max_pool(inputs))) 
        channel_attention = multiply([inputs, avg_out + max_out]) 
 
        # Spatial attention 
        avg_pool = tf.reduce_mean(channel_attention, axis=-1, keepdims=True) 
        max_pool = tf.reduce_max(channel_attention, axis=-1, keepdims=True) 
        spatial_attention = self.conv_spatial(tf.concat([avg_pool, max_pool], axis=-1)) 
 
        return multiply([channel_attention, spatial_attention])
    

class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = tf.keras.layers.Dense(embed_dim)  # Project each patch to embed_dim

    def call(self, images):
        batch_size = tf.shape(images)[0]
        height, width, channels = images.shape[1:]  # Ignore batch size, get the dimensions of the image
        
        patch_height, patch_width = self.patch_size
        # Extract patches (patches of shape [batch_size, num_patches_height, num_patches_width, patch_height * patch_width * channels])
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, patch_height, patch_width, 1],
            strides=[1, patch_height, patch_width, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        
        # Reshape patches to (batch_size, num_patches, patch_dim)
        num_patches_height = height // patch_height
        num_patches_width = width // patch_width
        patch_dim = patch_height * patch_width * channels  # Flattened patch size
        patches = tf.reshape(patches, [batch_size, num_patches_height * num_patches_width, patch_dim])
        
        # Project each patch to embed_dim using the Dense layer
        return self.projection(patches)  # [batch_size, num_patches, embed_dim]

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.positional_embedding = self.add_weight(
            shape=(1, num_patches, embed_dim),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, patch_embeddings):
        return patch_embeddings + self.positional_embedding

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

    def call(self, inputs):
        return self.attention(inputs, inputs)  # Self-attention

class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, hidden_dim):
        super(MLPBlock, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation="gelu")
        self.dense2 = tf.keras.layers.Dense(embed_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim):
        super(TransformerBlock, self).__init__()
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = MLPBlock(embed_dim, mlp_hidden_dim)

    def call(self, inputs):
        # Apply attention with residual connection
        attn_output = self.attention(self.norm1(inputs))
        x = inputs + attn_output

        # Apply MLP with residual connection
        mlp_output = self.mlp(self.norm2(x))
        return x + mlp_output ## NOTE the output size: (np, D)
    
class MyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()

    def call(self, x):
        # Example operation (replace this with your actual operation)
        # Ensuring that x is correctly processed as a Keras tensor
        return tf.reshape(x, (-1, 64, 64, 24))  # Example reshaping