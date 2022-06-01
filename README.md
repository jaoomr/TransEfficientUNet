# TransEfficientUNet in Keras

Architecture of a Transformer U-Net model with a EfficientNet in backbone. You can choose EfficientNet from B0 to B7. Implemented in Keras.

Transformer U-Net is based on TransUNet (https://arxiv.org/abs/2102.04306) concept. 
It consists in multiple convolutions + downsamples, a transformer encoder in the lowest 
resolution and a concatenation of upsample and previous convolutions. 

Here, the code uses a EfficientNet in downsample section.
