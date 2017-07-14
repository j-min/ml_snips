from scipy.misc import imread, imresize
import numpy as np

def image_read_and_resize(image_path, size=[224, 224]):
    """Read image from path and resize image into given size; [0, 255]"""
    
    im_ndarray = imread(image_path, mode='RGB')
    im_resized = imresize(im_ndarray, size=size)
    im_float32 = im_resized.astype(np.float32)
    return im_float32

def rescale(array):
    """[0, 255) => [-1, 1]"""
    array /= 255.
    array = np.clip(array, 0, 1)
    array -= 0.5
    array *= 2
    return array
