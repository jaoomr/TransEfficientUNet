import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import math
## EfficientNet Backbone Functions


def swish(x):
    return x * tf.nn.sigmoid(x)


def round_filters(filters, multiplier):
    depth_divisor = 8
    min_depth = None
    min_depth = min_depth or depth_divisor
    filters = filters * multiplier
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, multiplier):
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, ratio=0.25):
        super(SEBlock, self).__init__()
        self.num_reduced_filters = max(1, int(input_channels * ratio))
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.reduce_conv = tf.keras.layers.Conv2D(filters=self.num_reduced_filters,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same")
        self.expand_conv = tf.keras.layers.Conv2D(filters=input_channels,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same")

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = self.reduce_conv(branch)
        branch = swish(branch)
        branch = self.expand_conv(branch)
        branch = tf.nn.sigmoid(branch)
        output = inputs * branch
        return output


class MBConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate):
        super(MBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.conv1 = tf.keras.layers.Conv2D(filters=in_channels * expansion_factor,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                      strides=stride,
                                                      padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.se = SEBlock(input_channels=in_channels * expansion_factor)
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(rate=drop_connect_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = swish(x)
        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        x = self.se(x)
        x = swish(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.drop_connect_rate:
                x = self.dropout(x, training=training)
            x = tf.keras.layers.add([x, inputs])
        return x


def build_mbconv_block(in_channels, out_channels, layers, stride, expansion_factor, k, drop_connect_rate):
    block = tf.keras.Sequential()
    for i in range(layers):
        if i == 0:
            block.add(MBConv(in_channels=in_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=stride,
                             k=k,
                             drop_connect_rate=drop_connect_rate))
        else:
            block.add(MBConv(in_channels=out_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=1,
                             k=k,
                             drop_connect_rate=drop_connect_rate))
    return block


## CLASSIFIER

num_classes = 1
input_shape = (256, 256, 3)
patch_size = 1 
num_patches = 16 ** 2  
projection_dim = 256    #Embedding Vector dimmension
num_heads = 4   #Number of Attention Heads
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers


"""
## Implement multilayer perceptron (MLP)
"""


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tfa.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


"""
## Implement patch creation as a layer
"""


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


"""
## Implement the patch encoding layer

The `PatchEncoder` layer will linearly transform a patch by projecting it into a
vector of size `projection_dim`. In addition, it adds a learnable position
embedding to the projected vector.
"""


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


"""

## Build the Transformer U-Net model

Transformer U-Net is based on TransUNet (https://arxiv.org/abs/2102.04306) concept. 
It consists in multiple convolutions + downsamples, a transformer encoder in the lowest 
resolution and a concatenation of upsample and previous convolutions.


"""


#default values of width, depth and dropout = efficientnetB2 

def TransformerUNet(transformer_layers = 8, dropout=0.1, width_coefficient = 1.1, depth_coefficient = 1.2, drop_connect_rate = 0.3):

    inputs = layers.Input(shape=input_shape) #256x256 px
    #batches_input = layers.Lambda(lambda x: x/127.5 - 1)(inputs)
    conv1 = tf.keras.layers.Conv2D(filters=round_filters(32, width_coefficient),
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")(inputs)
    bn1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)(conv1)
    block1 = build_mbconv_block(in_channels=round_filters(32, width_coefficient),
                                     out_channels=round_filters(16, width_coefficient),
                                     layers=round_repeats(1, depth_coefficient),
                                     stride=1,
                                     expansion_factor=1, k=3, drop_connect_rate=drop_connect_rate)(bn1)
    block2 = build_mbconv_block(in_channels=round_filters(16, width_coefficient),
                                     out_channels=round_filters(24, width_coefficient),
                                     layers=round_repeats(2, depth_coefficient),
                                     stride=2,
                                     expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)(block1)
    block3 = build_mbconv_block(in_channels=round_filters(24, width_coefficient),
                                     out_channels=round_filters(40, width_coefficient),
                                     layers=round_repeats(2, depth_coefficient),
                                     stride=2,
                                     expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)(block2)
    block4 = build_mbconv_block(in_channels=round_filters(40, width_coefficient),
                                     out_channels=round_filters(80, width_coefficient),
                                     layers=round_repeats(3, depth_coefficient),
                                     stride=2,
                                     expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)(block3)
    block5 = build_mbconv_block(in_channels=round_filters(80, width_coefficient),
                                     out_channels=round_filters(112, width_coefficient),
                                     layers=round_repeats(3, depth_coefficient),
                                     stride=1,
                                     expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)(block4)

    bn2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)(block5)
    input_transformer = layers.Conv2D(256, (1, 1), padding='same' ,activation="relu", use_bias = False)(bn2)
    input_transformer = layers.BatchNormalization(epsilon=1e-6)(input_transformer)
    # Create patches.
    patches = Patches(patch_size)(input_transformer)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = tfa.layers.MultiHeadAttention(
            num_heads=num_heads, head_size=projection_dim, dropout=dropout
        )([x1, x1, x1])
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=dropout)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Reshape((16,16,projection_dim))(representation)
    representation = layers.Conv2D(256, (3, 3), padding='same' ,activation="relu", use_bias = False)(representation)
    representation = layers.BatchNormalization(epsilon=1e-6)(representation)


    upsample1 = layers.UpSampling2D(size=(2, 2))(representation) #32x32px
    skipConnectionAdd1 = layers.Concatenate()([upsample1, block3])
    decoder1 = layers.Conv2D(128, (3, 3), padding='same' ,activation="relu", use_bias = False)(skipConnectionAdd1)
    decoder1 = layers.BatchNormalization(epsilon=1e-6)(decoder1)

    upsample2 = layers.UpSampling2D(size=(2, 2))(decoder1) #64x64px
    skipConnectionAdd2 = layers.Concatenate()([upsample2, block2])
    skipConnectionAdd2 = layers.Dropout(dropout)(skipConnectionAdd2)
    decoder2 = layers.Conv2D(64, (3, 3), padding='same' ,activation="relu", use_bias = False)(skipConnectionAdd2)
    decoder2 = layers.BatchNormalization(epsilon=1e-6)(decoder2)

    upsample3 = layers.UpSampling2D(size=(2, 2))(decoder2) #128x128px
    skipConnectionAdd3 = layers.Concatenate()([upsample3, block1])
    skipConnectionAdd3 = layers.Dropout(dropout)(skipConnectionAdd3)
    decoder3 = layers.Conv2D(64, (3, 3), padding='same' ,activation="relu", use_bias = False)(skipConnectionAdd3)
    decoder3 = layers.BatchNormalization(epsilon=1e-6)(decoder3)

    upsample4 = layers.UpSampling2D(size=(2, 2))(decoder3) #256x256px
    decoder4 = layers.Conv2D(64, (3, 3), padding='same' ,activation="relu", use_bias = False)(upsample4)
    decoder4 = layers.BatchNormalization(epsilon=1e-6)(decoder4)

    output = layers.Conv2D(num_classes, (1, 1), padding='same' ,activation="sigmoid")(decoder4)
    #output = layers.Reshape((256*256, -1))(output)

    model = keras.Model(inputs=inputs, outputs=output)
    return model

# Params of different versions of efficientnet: width, depth, input_shape and drop_connect_rate, respectively

#def efficient_net_b0():
#   return get_efficient_net(1.0, 1.0, 224, 0.2)


#def efficient_net_b1():
#   return get_efficient_net(1.0, 1.1, 240, 0.2)


#def efficient_net_b2():
#    return get_efficient_net(1.1, 1.2, 256, 0.3)


#def efficient_net_b3():
#    return get_efficient_net(1.2, 1.4, 300, 0.3)


#def efficient_net_b4():
#    return get_efficient_net(1.4, 1.8, 380, 0.4)


#def efficient_net_b5():
#   return get_efficient_net(1.6, 2.2, 456, 0.4)


#def efficient_net_b6():
#    return get_efficient_net(1.8, 2.6, 528, 0.5)


#def efficient_net_b7():
    #return get_efficient_net(2.0, 3.1, 600, 0.5)

#vit = TransformerUNet()
#vit.summary()
#vit.load_weights("model_0_seed_2020.hdf5")