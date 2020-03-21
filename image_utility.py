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
from skimage import transform as tf
import os


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
        Returns:
            self.data [N_samples,N_pixels + ratio + label]; label -> float
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

    @property
    def show_stat(self):
        '''
        compute the statistics of the training data
        '''
        stats = {}
        sample_map = {}
        label_list = self.data[:, -1]

        for label in label_list:
            label = int(label)
            if label in sample_map:
                sample_map[label] += 1
            else:
                sample_map[label] = 1

        counter = []
        for key in sample_map:
            counter.append(sample_map[key])

        counter = np.array(counter).astype('float')

        stats = {'mean': np.mean(counter),
                 'median': np.median(counter),
                 'min': np.min(counter),
                 'max': np.max(counter),
                 'std': np.std(counter),
                 }

        return sample_map, stats

    def image_augmentation(self, mode, **kwargs):
        '''
        To augment some categories of the dataset

        Common method:
        rotation: random with angle between 0° and 360° (uniform)
        translation: random with shift between -10 and 10 pixels (uniform)
        rescaling: random with scale factor between 1/1.6 and 1.6 (log-uniform)
        flipping: yes or no (bernoulli)
        shearing: random with angle between -20° and 20° (uniform)
        stretching: random with stretch factor between 1/1.3 and 1.3 (log-uniform)

        Args:
            kwargs['shear_param'] ~ 0.1, 0.2
            kwargs['translate_param'] (1,2) -> left shift 1, up shift 2
        '''

        sample_map, stats = self.show_stat

        if mode == 1:
            '''
            mode 1 do full augmentation:

            label whose samples below the average number will be augmented
            '''
            aug_label = [float(key) for key in sample_map if float(
                sample_map[key]) <= float(stats['mean'])]

            for image in self.data:
                if image[-1] in aug_label:
                    try:
                        aug_list = np.concatenate(
                            [aug_list, self._flip_image(image).reshape(1, -1)], axis=0)
                        aug_list = np.concatenate(
                            [aug_list, self._rotate(image).reshape(1, -1)], axis=0)
                        aug_list = np.concatenate(
                            [aug_list, self.shear_image(image, kwargs['shear_param_1']).reshape(1, -1)], axis=0)
                        aug_list = np.concatenate(
                            [aug_list, self.translate_image(image, kwargs['translate_param_1']).reshape(1, -1)], axis=0)

                    except NameError:
                        aug_list = self._flip_image(image).reshape(1, -1)
                        aug_list = np.concatenate(
                            [aug_list, self._rotate(image).reshape(1, -1)], axis=0)
                        aug_list = np.concatenate(
                            [aug_list, self.shear_image(image, kwargs['shear_param_1']).reshape(1, -1)], axis=0)
                        aug_list = np.concatenate(
                            [aug_list, self.translate_image(image, kwargs['translate_param_1']).reshape(1, -1)], axis=0)

        elif mode == 2:
            '''
            mode 2: only do shear and translation, according to the second set parameters
            '''
            aug_label = [float(key) for key in sample_map if float(
                sample_map[key]) <= float(stats['mean'])]

            for image in self.data:
                if image[-1] in aug_label:
                    try:
                        aug_list = np.concatenate(
                            [aug_list, self.shear_image(image, kwargs['shear_param_2']).reshape(1, -1)], axis=0)
                        aug_list = np.concatenate(
                            [aug_list, self.translate_image(image, kwargs['translate_param_2']).reshape(1, -1)], axis=0)

                    except NameError:

                        aug_list = self.shear_image(
                            image, kwargs['shear_param_2']).reshape(1, -1)

                        aug_list = np.concatenate(
                            [aug_list, self.translate_image(image, kwargs['translate_param_2']).reshape(1, -1)], axis=0)

        self.data = np.concatenate([self.data, aug_list], axis=0)

    def _flip_image(self, image):
        '''
        Args:
            image -> np 1d array
            the first 25 * 25 numbers are the pixel of the image

        Output:
            [25*25 + ratio + label] (the flipped image)
        '''
        img = image.ravel()[:25*25].reshape(25, 25)
        return np.append(np.flip(img, axis=1).ravel(), image.ravel()[25*25:])

    def _rotate(self, image):
        '''
        Args:
            image -> np 1d array
            the first 25 * 25 numbers are the pixel of the image

        Output:
            [25*25 + ratio + label] (the rotated image)
        '''
        img = image.ravel()[:25*25].reshape(25, 25)
        return np.append(img.T.ravel(), image.ravel()[25*25:])

    def shear_image(self, image, radian):
        '''
        Args:
            radian ~ 0.1, 0.2
        '''

        img = image.ravel()[:25*25].reshape(25, 25)

        affine_tf = tf.AffineTransform(rotation=radian)

        return np.append(tf.warp(img, inverse_map=affine_tf).ravel(), image.ravel()[25*25:])

    def translate_image(self, image, Coe_translation: tuple):
        '''
        Args:
            Coe_translation (1,2) -> left shift 1, up shift 2
        '''

        img = image.ravel()[:25*25].reshape(25, 25)

        affine_tf = tf.AffineTransform(translation=Coe_translation)

        return np.append(tf.warp(img, inverse_map=affine_tf).ravel(), image.ravel()[25*25:])

    def shuffler(self):
        '''
        to shuffle the data [N_sample,pixel + ratio + label]
        '''
        np.random.shuffle(self.data)

    @staticmethod
    def batcher(x, y, batch_size):
        l = len(y)
        for batch in range(0, l, batch_size):
            yield (x[batch:min(batch + batch_size, l)], y[batch:min(batch + batch_size, l)])

    def save_npy(self, path):
        '''
            Args:
                path: the file path you want to save, exclude file name

                image -> np matrix; (N_sample,625)
                mum_label -> float, (N_sample,)

        '''
        images = self.data[:, :-2]
        num_labels = self.data[:, -1]

        for i, img in enumerate(images):
            num_label = int(num_labels[i])

            # get the class name
            for key, value in self.label2num.items():
                if value == num_label:
                    label = key

            class_name = label

        # root/class_x/xxx.ext

            file_name = os.path.join(path, f'{class_name}/sample{i}.npy')

            np.savetxt(fname=file_name, X=img, fmt='%.18e', delimiter=' ', newline='\n',
                       header='', footer='', comments='# ', encoding=None)
        pass
