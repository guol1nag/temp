import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sklearn.model_selection as sk_ModelSelection
import copy
import torch
import time
import seaborn as sns
import zipfile
from PIL import Image
from pylab import cm
from skimage import segmentation
from skimage.morphology import watershed
from skimage import measure
from skimage import morphology
from skimage.io import imread
from skimage.transform import resize


class Dataset_utility():
    def __init__(self, train, label, label2num):
        '''
        Args:
            train: a list of train samples, each element is a graph but could have different shape
            label: a list of corresponding label sampless
            label2num: a map from label name to label number, (first number, second number) = (label category, label frequency)
        '''
        self.train = train
        self.label = label
        self.label2num = label2num
        self.y = list()
        for l in label:
            self.y.append(self.label2num[l][0])  # first num is category
        self.y = np.array(self.y)
        # get the total training images

    def counter(self):
        '''
        count how many images corresponding to each label
        '''
        counter = {}
        for label in self.y:
            if label in counter:
                counter[label] += 1
            else:
                counter[label] = 1
        return counter

    def _getMinorMajorRatio(self, image):
        image = image.copy()
        # thresholded image
        imagethr = np.where(image > np.mean(image), 0., 1.0)

        # Dilate the image
        imdilated = morphology.dilation(imagethr, np.ones((4, 4)))

        # Create the label list
        label_list = measure.label(imdilated)
        label_list = imagethr*label_list
        label_list = label_list.astype(int)

        region_list = measure.regionprops(label_list)
        maxregion = self._getLargestRegion(region_list, label_list, imagethr)

        # guard against cases where the segmentation fails by providing zeros
        ratio = 0.0
        if ((not maxregion is None) and (maxregion.major_axis_length != 0.0)):
            ratio = 0.0 if maxregion is None else maxregion.minor_axis_length * \
                1.0 / maxregion.major_axis_length
        return ratio

    def _getLargestRegion(self, props, labelmap, imagethres):
        regionmaxprop = None
        for regionprop in props:
            # check to see if the region is at least 50% nonzero
            if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
                continue
            if regionmaxprop is None:
                regionmaxprop = regionprop
            if regionmaxprop.filled_area < regionprop.filled_area:
                regionmaxprop = regionprop
        return regionmaxprop

    def image_rescaling(self):
        '''
        Args:
            -> self.data [N_samples,N_pixels + ratio + label] 
        '''

        number_of_Images = len(self.train)
        num_rows = number_of_Images  # one row for each image in the training dataset

        # We'll rescale the images to be 25x25
        maxPixel = 25
        imageSize = int(maxPixel * maxPixel)
        num_features = imageSize + 2  # for our ratio and label

        # data is the feature vector with one row of features per image
        # consisting of the pixel values and our metric
        self.data = np.zeros((num_rows, num_features), dtype=float)

        print('start rescaling')
        # Navigate through the list of directories
        for c, image in enumerate(self.train):
            # image = imread(nameFileImage, as_grey=True)
            # files.append(nameFileImage)
            axisratio = self._getMinorMajorRatio(image)
            image = resize(image, (maxPixel, maxPixel))

            # Store the rescaled image pixels and the axis ratio
            self.data[c, 0:imageSize] = image.reshape(1, imageSize)
            self.data[c, imageSize] = axisratio
            self.data[c, imageSize+1] = self.y[c]

    def shuffler(self):
        '''
        Outputs:
            self.data (after shuffle)
        '''
        pass

    def image_augmentation(self):
        '''
        rotation: random with angle between 0° and 360° (uniform)
        translation: random with shift between -10 and 10 pixels (uniform)
        rescaling: random with scale factor between 1/1.6 and 1.6 (log-uniform)
        flipping: yes or no (bernoulli)
        shearing: random with angle between -20° and 20° (uniform)
        stretching: random with stretch factor between 1/1.3 and 1.3 (log-uniform)
        '''
        pass

    def _flip_image(self, image):
        '''
        Args:
            image -> np 1d array 
            the first 25 * 25 numbers are the pixel of the image

        Output:
            [25*25 + ratio + label] (the flipped image)
        '''
        img = image.ravel()[:25*25].reshape(25, 25)
        return np.flip(img, axis=1).ravel() + image.ravel()[25*25:]

    def _rotate(self, image):
        '''
        Args:
            image -> np 1d array 
            the first 25 * 25 numbers are the pixel of the image

        Output:
            [25*25 + ratio + label] (the rotated image)
        '''
        img = image.ravel()[:25*25].reshape(25, 25)
        return img.T.ravel() + image.ravel()[25*25:]
