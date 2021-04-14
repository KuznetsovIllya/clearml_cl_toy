#!/usr/bin/env python
# coding: utf-8

# ## Adding background for masked images, stage1 for masked images augmentation

import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import sys
from os import listdir, scandir
from os.path import isfile, join, basename, dirname
import cv2
from matplotlib import pyplot as plt
import pycocotools
import json
from pycocotools import mask as cmask
from skimage import measure
import base64
from IPython.core.debugger import set_trace
from random import sample as rnd_sample, randrange
from math import ceil
from bs4 import BeautifulSoup as Soup
from xml.dom import minidom
import numpy as np
import shutil
from clearml import Task
import pathlib

task = Task.init(project_name='rabbit_fox', task_name='image_augmentation')


# setting paths and variables
obj_images_dir = 'source/'
subfolders = [ basename(f.path) for f in scandir(obj_images_dir) if f.is_dir()]
augment_images_dir = 'destination/'
name_uniq = '_1003' #TO-DO - change to current date
name_counter = 0
percentage_geo_transf = 10 


# defining augmentations:
# defining non-geometrical augmenters
au_gauss_noise = iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))
au_blend_alpha = iaa.BlendAlpha((0.1, 0.4), iaa.Grayscale(1.0))
au_aver_blur = iaa.AverageBlur(k=(2, 4))
au_shot_noise = iaa.imgcorruptlike.ShotNoise(severity=1)

au_jpeg_comp = iaa.JpegCompression(compression=(50, 90))
au_change_col_temp = iaa.ChangeColorTemperature((4420, 7660))
au_gamma1 = iaa.GammaContrast(0.7, 0.7)
au_gamma2 = iaa.GammaContrast(0.7, 0.8)
au_gamma3 = iaa.GammaContrast(0.7, 0.9)
au_linear_contrast = iaa.LinearContrast((1.0, 1.3))
au_horiz_grad1 = iaa.BlendAlphaHorizontalLinearGradient(iaa.TotalDropout(1.0), min_value=0.0, max_value=0.7)
au_horiz_grad2 = iaa.BlendAlphaHorizontalLinearGradient(iaa.TotalDropout(1.0), min_value=0.7, max_value=0.0)
au_horiz_grad3 = iaa.BlendAlphaHorizontalLinearGradient(iaa.TotalDropout(1.0), min_value=0.3, max_value=0.6)
au_vert_grad1 = iaa.BlendAlphaVerticalLinearGradient(iaa.TotalDropout(1.0), min_value=0.0, max_value=0.7)
au_vert_grad2 = iaa.BlendAlphaVerticalLinearGradient(iaa.TotalDropout(1.0), min_value=0.7, max_value=0.0)
au_vert_grad3 = iaa.BlendAlphaVerticalLinearGradient(iaa.TotalDropout(1.0), min_value=0.3, max_value=0.6)
au_bright_chan = iaa.WithBrightnessChannels(iaa.Add((-28, 24)))
au_mult_bright = iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.0), add=(-30, 30))
# au_persp_transform = iaa.PerspectiveTransform(scale=(0.01, 0.05))
au_seq1 = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
    iaa.BlendAlpha((0.1, 0.4), iaa.Grayscale(1.0))
    ])
au_seq2 = iaa.Sequential([
    iaa.BlendAlpha((0.1, 0.4), iaa.Grayscale(1.0)),
    iaa.AverageBlur(k=(2, 4))
    ])
au_seq3 = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
    iaa.BlendAlpha((0.1, 0.4), iaa.Grayscale(1.0)),
    iaa.imgcorruptlike.ShotNoise(severity=1)
    ])
au_seq4 = iaa.Sequential([
    iaa.BlendAlpha((0.1, 0.4), iaa.Grayscale(1.0)),
    iaa.AverageBlur(k=(2, 4)),
    iaa.imgcorruptlike.ShotNoise(severity=1)
    ])
au_salt_perrer_channel = iaa.SaltAndPepper(0.05, per_channel=True)

au_list_nongeo = [au_gauss_noise, au_blend_alpha, au_aver_blur, au_shot_noise, au_jpeg_comp, au_change_col_temp, au_gamma1, au_gamma2, au_gamma3, au_linear_contrast, au_horiz_grad1, au_horiz_grad2, au_horiz_grad3, au_vert_grad1, au_vert_grad2, au_vert_grad3, au_bright_chan, au_mult_bright, au_seq1, au_seq2, au_seq3, au_seq4, au_salt_perrer_channel]

rep_list_nongeo = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

# defining non-geometrical augmenters
au_scalex = iaa.ScaleX((0.4, 1.3))
au_scaley = iaa.ScaleY((1, 2))
au_rotate = iaa.Affine(rotate =(-10, 10))
au_shearx = iaa.ShearX((-7, 7))
au_sheary = iaa.ShearY((-7, 7)) 
au_persp_transform = iaa.PerspectiveTransform(scale=(0.01, 0.05))

# list of augmenters
au_list_geo = [au_scalex, au_scaley, au_rotate, au_shearx, au_sheary, au_persp_transform]
# list of how many times each augmenter should be run; the number of entries in list should correspond to number
# of entries in au_list
rep_list_geo = [1,1,1,1,1,1]

# obj_images = rnd_sample(obj_images_orig, ceil(len(obj_images_orig)*(percentage_geo_transf/100)))

for subfld in subfolders:
    curr_dir = obj_images_dir + '/' + subfld
    pathlib.Path(augment_images_dir + '/' + 'train' + '/' +subfld).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(augment_images_dir + '/' + 'eval' + '/' +subfld).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(augment_images_dir + '/' + 'test' + '/' +subfld).mkdir(parents=True, exist_ok=True) 

    obj_images = [f for f in listdir(curr_dir) if (isfile(join(curr_dir, f)) and (f.endswith('.jpg')))]

    for img_name in obj_images:
        try:
            dice = randrange(1,101)
            if dice < 81:
                split_folder = 'train'
            elif dice < 91:
                split_folder = 'eval'
            else:
                split_folder = 'test'
            # FIRST TRANSFORMATION STAGE
            name_counter += 1

            # reading image with objects file and backround image file
            img = cv2.imread(curr_dir + '/' + img_name)
            img_shape = img.shape
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # SECOND (NON_GEO) TRANSFROMATION STAGE
            print("INFO: STARTING BASIC TRANFORMATIONS")

            for aug_count, augmenter in enumerate(au_list_nongeo):
                print('Augmenter started: ', augmenter)
                print('aug_count __________________________________________________________ ', aug_count)
                for augnum in range(1,rep_list_nongeo[aug_count]+1):
                    print('Augmentation cycle: {}'.format(augnum))
                    print('augnum __________________________________________________________ ', augnum)
                    try:

                        #augmenting the image
                        image_aug = augmenter(image = img)




                        # randomly selecting images for geometric augmentation 
                        rand_geo_transf = randrange(1,101)
                        if rand_geo_transf >  percentage_geo_transf:
                            dice = randrange(1,101)
                            if dice < 81:
                                split_folder = 'train/'
                            elif dice < 91:
                                split_folder = 'eval/'
                            else:
                                split_folder = 'test/'
                            img_nongeo_name = augment_images_dir + split_folder + subfld + '/' + str(name_counter) + '_' + str(aug_count) + '_x_' + name_uniq
                            imageio.imwrite(img_nongeo_name + '.jpg', image_aug)
                            # read saved augmented image and convert it to base64 format

                        else:

                            # THIRD TRANSFORMATION STAGE
                            print("INFO: STARTING GEOMETRIC TRANSFORMATIONS")


                            for aug_count_geo, augmenter_geo in enumerate(au_list_geo):
                                print('Geometric augmenter started: ', augmenter_geo)
                                for augnum_geo in range(1,rep_list_geo[aug_count_geo]+1):
                                    print('Geometric augmentation cycle: {}'.format(augnum_geo))
                                    try:


                                        image_aug_geo = augmenter_geo(image=image_aug)
                                        # getting numpy array from augmented mask

                                        dice = randrange(1,101)
                                        if dice < 81:
                                            split_folder = 'train/'
                                        elif dice < 91:
                                            split_folder = 'eval/'
                                        else:
                                            split_folder = 'test/'

                                        aug_image_full_name = augment_images_dir  + split_folder + subfld + '/' +  str(name_counter) + '_' + str(aug_count) + '_' + str(aug_count_geo) + name_uniq + '.jpg'
                                        imageio.imwrite(aug_image_full_name, image_aug_geo)

                                    
                                    except Exception as e:
                                        print("ERROR: Third (geo) stage failed!")
                                        print(e)
                                        exit()


                    except Exception as e:
                        print("ERROR: Second (non-geo) stage failed!")
                        print(e)



        except Exception as e:
            print('shit!')
            print(e)
