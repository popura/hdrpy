import cv2
import numpy as np
import argparse
import sys
# import hdrpy
import math
from cv2.ximgproc import guidedFilter
# from core.filter import GuidedFilter
from gf import guided_filter

class MertensFusion:
    def __init__(self):
        pass

    def fuse(self, images):
        merge_mertens = cv2.createMergeMertens()
        return merge_mertens.process(images)

class NejatiFusion:
    def __init__(self):
        self.r = 12
        self.eps = 0.25
        self.sigma_D = 0.12
        self.sigma_l = 0.5
        self.sigma_g = 0.2
        self.alpha = 1.1

    def fuse(self, images):
        baselayer_list = []
        detaillayer_list = []
        base_weight_list = []
        detail_weight_list = []

        for i, img in enumerate(self.img_list):
            img = img / 255
            img = img.astype(np.float32)

            [base, detail] = self.decompose(img)
            baselayer_list.append(base)
            detaillayer_list.append(detail)

            base_weight = self.weight_base(img, base)
            base_weight_list.append(base_weight)

            detail_weight = self.weight_detail(lum)
            detail_weight_list.append(detail_weight)

        fused_base = self.fuse_base(baselayer_list, base_weight_list)
        fused_detail = self.fuse_detail(detaillayer_list, detail_weight_list)

        return self.compose(fused_base, fused_detail)

    # decompose an image into a base layer and a detail layer
    def decompose(self, image):
        lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        base = guided_filter(lum, lum, self.r, self.eps)
        detail = np.zeros(self.img_list[0].shape)
        for i in range(3):
            detail[:, :, i] = img[:, :, i] - baselayer
        return [base, detail]

    # combine a base layer with a detail layer
    def compose(self, base, detail):
        image = np.zeros(detail.shape)
        for i in range(3):
            image[:, :, i] = base + (self.alpha * detail[:, :, i])
        return image

    # calculate weights for a base layer
    def weight_base(self, image, base):
        lum = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        weight_B_l = np.exp(-((base - 0.5) ** 2) / (2 * (self.sigma_l ** 2)))
        weight_B_g = np.exp(-((lum.mean() - 0.5) ** 2) / (2 * (self.sigma_g ** 2)))
        return weight_B_l * weight_B_g

    # calculate weights for a detail layer
    def weight_detail(self, image):
        lum = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fai_D = cv2.blur(lum, (7, 7))
        return np.exp(-((fai_D - 0.5) ** 2) / (2 * (self.sigma_D ** 2)))

    def fuse_base(self, base_layers, weights):
        normalized_weights = self.normalize_weights(weights)
        fused_base = np.zeros(base_layers[0].shape)
        for base, weight in zip(base_layers, normalized_weights):
            fused_base += weight * base
        return fused_base

    def fuse_detail(self, detail_layers, weights):
        normalized_weights = self.normalize_weights(weights)
        fused_detail = np.zeros(detail_layers[0].shape)
        for detail, weight in zip(detail_layers, normalized_weights):
            fused_detail += weight * detail
        return fused_detail

    def normalize_weights(self, weights):
        total_weight = np.zeros(weights[0].shape)
        normalized_weights = []
        for weight in weights:
            total_weight += weight
        for weight in weights:
            normalized_weights.append(weight / total_weight)
        return normalized_weights
