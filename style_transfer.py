import os

try:
    import cPickle as pickle
except:
    import pickle

import cv2 as cv
import numpy as np
from face_alignment import FaceAlignment, LandmarksType
from skimage.io import imread, imsave, imshow

from image_morpher import ImageMorpher


class StyleTransfer:
    def __init__(self, style_img, input_img, style_mask, input_mask, save=False):
        style_name = os.path.basename(style_img).split('.')[0]
        input_name = os.path.basename(input_img).split('.')[0]

        self.style_img = np.float32(imread(style_img))
        self.input_img = np.float32(imread(input_img))

        self.style_mask = np.float32(imread(style_mask))
        self.input_mask = np.float32(imread(input_mask))

        # Fetch Facial Landmarks
        if os.path.exists('input/%s_%s_lm.pkl' % (style_name, input_name)):
            with open('input/%s_%s_lm.pkl' % (style_name, input_name), 'rb') as f:
                pkl = pickle.load(f)
                self.style_lm = pkl['style']
                self.input_lm = pkl['input']
        else:
            fa = FaceAlignment(LandmarksType._2D, device='cpu', flip_input=False)
            self.style_lm = fa.get_landmarks(self.style_img)[0]
            self.input_lm = fa.get_landmarks(self.input_img)[0]
            with open('input/%s_%s_lm.pkl' % (style_name, input_name),
                      'wb') as f:
                pickle.dump({
                    'style': self.style_lm,
                    'input': self.input_lm
                }, f, protocol=2)

        self.output_filename = '_'.join({input_name, style_name})
        self.save = save

    def run(self):
        warped, vx, vy = self.dense_matching(self.style_img, self.input_img, self.style_lm, self.input_lm)
        matched = self.local_matching(self.style_img, self.input_img, self.style_mask, self.input_mask, vx, vy)
        matched = self.replace_bkg(matched, self.style_img, self.input_img, self.style_mask, self.input_mask, vx, vy)
        matched = self.eye_highlight(matched, self.style_img, self.input_img, self.style_mask, self.input_mask, vx, vy)

        if self.save:
            imsave(self.output_filename + '.jpg', matched)

        return

    @staticmethod
    def dense_matching(style, input, style_lm, input_lm):
        """
        Warp image using landmarks:
        Uses feature based image metamorphosis,
        Dense SIFT Flow
        :return: nd array of warped image
        """

        im = ImageMorpher()
        morphed_img = im.run(style, input, style_lm, input_lm)

        # Match better with Dense SIFT

        return morphed_img

    @staticmethod
    def local_matching(style, input, style_mask, input_mask, vx, vy):
        h, w, c = input.shape
        new_h, new_w, = h, w,
        new_style, new_input = np.copy(style), np.copy(input)
        new_style[style_mask == 0] = 0
        new_input[input_mask == 0] = 0
        n_stacks = 7

        # Build a Laplacian Stack
        laplace_style = []
        laplace_input = []
        for i in range(n_stacks):
            new_h, new_w = int(new_h / 2), int(new_w / 2)
            new_style = cv.pyrDown(new_style, np.zeros((new_h, new_w, c)))
            new_input = cv.pyrDown(new_input, np.zeros((new_h, new_w, c)))
            if i is 0:
                laplace_style.append(style - cv.resize(new_style, (w, h)))
                laplace_input.append(input - cv.resize(new_input, (w, h)))
            else:
                temp_style = cv.resize(pre_style, (w, h)) - cv.resize(new_style, (w, h))
                temp_input = cv.resize(pre_input, (w, h)) - cv.resize(new_input, (w, h))
                laplace_style.append(temp_style)
                laplace_input.append(temp_input)

            pre_style = new_style
            pre_input = new_input

        resid_style = cv.resize(new_style, (w, h))
        resid_input = cv.resize(new_input, (w, h))

        # Compute Local Energies
        # Power maps, Malik and Perona 1990
        energy_style = []
        energy_input = []
        for i in range(n_stacks):
            new_style_ener = cv.pyrDown(laplace_style[i] ** 2, (new_h, new_w, c))
            new_input_ener = cv.pyrDown(laplace_input[i] ** 2, (new_h, new_w, c))

            for j in range(i - 1):
                new_style_ener = cv.pyrDown(new_style_ener, (new_h, new_w, c))
                new_input_ener = cv.pyrDown(new_input_ener, (new_h, new_w, c))

            energy_style.append(cv.resize(np.sqrt(new_style_ener), (w, h)))
            energy_input.append(cv.resize(np.sqrt(new_input_ener), (w, h)))

        # Post-process warping style stacks:
        for i in range(len(energy_style)):
            laplace_style[i] = laplace_style[i][vy, vx]
            energy_style[i] = energy_style[i][vy, vx]

        # Compute Gain Map and Transfer
        eps = 0.01 ** 2
        gain_max = 2.8
        gain_min = 0.005
        output = np.zeros((h, w, c))
        for i in range(n_stacks):
            gain = np.sqrt(np.divide(energy_style[i], (energy_input[i] + eps)))
            gain[gain <= gain_min] = 1
            gain[gain > gain_max] = gain_max
            output += np.multiply(laplace_input[i], gain)
        output += resid_style

        return output

    @staticmethod
    def replace_bkg(matched, style, input, style_mask, input_mask, vx, vy):
        temp = np.zeros(input.shape, dtype=np.uint8)
        temp[style_mask == 0] = style[style_mask == 0]
        temp[input_mask == 255] = 0
        # xy = (255 - style_mask).astype(np.uint8)
        # bkg = cv.inpaint(temp, xy[:, :, 0], 10, cv.INPAINT_TELEA)
        # imsave('output/bkg.jpg', bkg.astype(int))
        # TODO: Extrapolate background
        xy = np.logical_not(input_mask.astype(bool))
        matched[xy] = 0
        output = temp + matched
        output[output > 255] = 255
        output[output <= 0] = 0
        output = output.astype(int)
        imsave('output/temp.jpg', output)
        # imsave('output/temp.jpg', style.astype(int))
        return matched

    @staticmethod
    def eye_highlight(matched, style, input, style_mask, input_mask, vx, vy):
        return matched
