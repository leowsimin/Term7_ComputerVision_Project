import tensorflow as tf
from tensorflow.python.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda, Layer


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


## Implementing Vit Transformer 

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
        patch_dim = patch_height * patch_width * channels # Flattened patch size
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
        print("Patch Emdding:", patch_embeddings)
        print("Positional Embedding:", self.positional_embedding)
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
    
# class MyLayer(tf.keras.layers.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()

#     def call(self, x):
#         # Example operation (replace this with your actual operation)
#         # Ensuring that x is correctly processed as a Keras tensor
#         return tf.reshape(x, (-1, 64, 64, 24))  # Example reshaping
    
# class ReshapeLayer(Layer):
#     def __init__(self, target_shape, **kwargs):
#         super(ReshapeLayer, self).__init__(**kwargs)
#         self.reshape = Reshape(target_shape)

#     def call(self, inputs):
#         return self.reshape(inputs)

# class ReshapeLayer(Layer):
#     def __init__(self, target_shape, **kwargs):
#         super(ReshapeLayer, self).__init__(**kwargs)
#         self.target_shape = target_shape

#     def call(self, inputs):
#         return K.reshape(inputs, self.target_shape) 