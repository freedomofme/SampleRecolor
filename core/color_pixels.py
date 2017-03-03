# -*- coding: utf-8 -*-

import numpy as np

from cv.image import to32F, rgb2Lab, rgb2hsv, gray2rgb


## Implementation of color pixels.
#
#  input image is automatically converted into np.float32 format.
class ColorPixels:
    ## Constructor
    #  @param image          input image.
    #  @param num_pixels     target number of pixels from the image.
    def __init__(self, image, num_pixels=1000):
        self._image = to32F(image)
        self._num_pixels = num_pixels
        self._rgb_pixels = None
        self._Lab = None
        self._hsv = None

    ## RGB pixels.
    def rgb(self):
        if self._rgb_pixels is None:
            self._rgb_pixels = self.pixels("rgb")
        return self._rgb_pixels

    ## Lab pixels.
    def Lab(self):
        if self._Lab is None:
            self._Lab = self.pixels("Lab")
        return self._Lab

    ## HSV pixels.
    def hsv(self):
        if self._hsv is None:
            self._hsv = self.pixels("hsv")
        return self._hsv

    ## Pixels of the given color space.
    def pixels(self, color_space="rgb", all = False):
        image = np.array(self._image)
        if color_space == "rgb":
            if _isGray(image):
                image = gray2rgb(image)

        if color_space == "Lab":
            image = rgb2Lab(self._image)
            # print image

        if color_space == "hsv":
            image = rgb2hsv(self._image)

        if all == True:
            return self._image2pixels(image, s = 1)

        return self._image2pixels(image)

    #此处下采样
    def _image2pixels(self, image, s = 0):
        if _isGray(image):
            h, w = image.shape
            step = h * w / self._num_pixels
            if s == 1:
                return image.reshape((h * w))[::s]
            return image.reshape((h * w))[::step]

        h, w, cs = image.shape
        step = h * w / self._num_pixels

        if s == 1:
            return image.reshape((-1, cs))[::s]
        return image.reshape((-1, cs))[::step]


def _isGray(image):
    return len(image.shape) == 2
