import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

def upsample(filter):
    result = keras.Sequential(name='UpSample_'+str(filter))

    result.add(layers.DepthwiseConv2D(5, padding='same', use_bias=False))
    result.add(layers.BatchNormalization())
    result.add(layers.ReLU())
    result.add(layers.Conv2D(filter, 1, padding='valid',  use_bias=False))
    result.add(layers.BatchNormalization())
    result.add(layers.ReLU())
    result.add(layers.UpSampling2D(size=(2, 2), interpolation='nearest'))

    return result

def Generator():
    Mobilnet = tf.keras.applications.MobileNet(weights="imagenet")

    up_stack = [
        upsample(512),       # (1, 14, 14, 512)
        upsample(256),       # (1, 28, 28, 256)
        upsample(128),       # (1, 56, 56, 128)
        upsample(64),        # (1, 112, 112, 64)
        upsample(32),        # (1, 224, 224, 32)
    ]

    last = keras.Sequential([
        layers.Conv2D(1, 1, padding='valid',  use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU()
    ],name='Last_Layer'
    )


    ## Forward ##
    x = Mobilnet.input

    c=0
    skip_count = 0
    s=[] # Skip Layers
    for layer in Mobilnet.layers:
        if c != 0 and c < 87:
            x=layer(x)
            if type(layer) == layers.ReLU:
                skip_count += 1
                if skip_count == 3 or skip_count == 5 or skip_count == 9:
                    s.append(x)
        c += 1

    skip_count = 0
    for up in up_stack:
        skip_count += 1
        x=up(x)
        if skip_count == 2:
            x= x+s[2]
        if skip_count == 3:
            x= x+s[1]
        if skip_count == 4:
            x= x+s[0]

    last = last(x)

    return Model(inputs=Mobilnet.input, outputs=last)
