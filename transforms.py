import h5py  #The h5py package is a interface to the HDF5 binary data format.
import numpy as np
import tensorflow as tf


def resize(inimg, tgimg, height, width):

    inimg = tf.image.resize(np.rollaxis(inimg,0,3), [height, width], method='nearest')
    tgimg = tf.image.resize(np.expand_dims(tgimg, axis=2), [height, width],method='nearest')

    return inimg, tgimg

def random_jitter(inimg,tgimg):
    inimg, tgimg = resize(inimg, tgimg, 280,280)

    inimg = tf.image.random_crop(inimg, size=[224,224,3],seed=23)
    tgimg = tf.image.random_crop(tgimg, size=[224,224,1],seed=23)

    if  tf.random.uniform(()) > 0.5:
        inimg = tf.image.flip_left_right(inimg)
        tgimg = tf.image.flip_left_right(tgimg)
    return inimg, tgimg


def load_image(filename, augment=True):
    hf = h5py.File(filename.numpy(), 'r')
    image = hf['rgb'][...]
    label = hf['depth'][...]

    #image = tf.cast(hf['rgb'][...],tf.float32)
    #label = tf.cast(hf['depth'][...], tf.float32)

    if augment:
        image,label = random_jitter(image,label)
    else:
        image,label = resize(image,label,224,224)
    return image, label

def load_train_image(filename):
    return load_image(filename,True)

def load_test_image(filename):
    return load_image(filename,False)
