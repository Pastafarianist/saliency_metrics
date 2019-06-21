#!/usr/bin/env python3

"""
This implementation is based on
* https://github.com/tarunsharma1/saliency_metrics (Python version)
* https://github.com/cvzoya/saliency/tree/master/code_forMetrics (original Matlab code)
"""

import math
import random
import sys

import matplotlib.pyplot as plt
import numpy as np


def generate_dummy(size=14, num_fixations=100, num_saliency_points=200):
    # first generate dummy fixation_map and saliency map
    discrete_gt = np.zeros((size, size))
    saliency_map = np.zeros((size, size))

    for i in range(0, num_fixations):
        discrete_gt[np.random.randint(size), np.random.randint(size)] = 1.0

    for i in range(0, num_saliency_points):
        saliency_map[np.random.randint(size), np.random.randint(size)] = 255 * round(random.random(), 1)

    # check if fixation_map and saliency_map are same size
    assert discrete_gt.shape == saliency_map.shape, 'sizes of ground truth and saliency map don\'t match'
    return saliency_map, discrete_gt


def min_max_scale(saliency_map):
    # normalize the saliency map (as done in MIT code)
    s_min, s_max = np.min(saliency_map), np.max(saliency_map)
    return (saliency_map - s_min) / (s_max - s_min)


def standard_scale(saliency_map):
    return (saliency_map - np.mean(saliency_map)) / np.std(saliency_map)


def discretize_gt(fixation_map):
    import warnings
    warnings.warn('can improve the way GT is discretized')
    return fixation_map / 255


def auc_judd(saliency_map, fixation_map):
    # ground truth is discrete, saliency_map is continuous and normalized
    fixation_map = discretize_gt(fixation_map)
    # thresholds are calculated from the saliency map, only at places where fixations are present
    thresholds = []
    for i in range(0, fixation_map.shape[0]):
        for k in range(0, fixation_map.shape[1]):
            if fixation_map[i][k] > 0:
                thresholds.append(saliency_map[i][k])

    num_fixations = np.sum(fixation_map)
    # num fixations is no. of saliency map values at fixation_map >0

    thresholds = sorted(set(thresholds))

    # fp_list = []
    # tp_list = []
    area = []
    area.append((0.0, 0.0))
    for thresh in thresholds:
        # in the saliency map, keep only those pixels with values above threshold
        temp = np.zeros(saliency_map.shape)
        temp[saliency_map >= thresh] = 1.0
        assert np.max(
            fixation_map) == 1.0, 'something is wrong with ground truth..not discretized properly max value > 1.0'
        assert np.max(
            saliency_map) == 1.0, 'something is wrong with saliency map..not normalized properly max value > 1.0'
        num_overlap = np.where(np.add(temp, fixation_map) == 2)[0].shape[0]
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of pixels that overlap with fixation_map / total number of non fixated pixels
        # this becomes nan when fixation_map is full of fixations..this won't happen
        fp = (np.sum(temp) - num_overlap) / ((np.shape(fixation_map)[0] * np.shape(fixation_map)[1]) - num_fixations)

        area.append((round(tp, 4), round(fp, 4)))
    # tp_list.append(tp)
    # fp_list.append(fp)

    # tp_list.reverse()
    # fp_list.reverse()
    area.append((1.0, 1.0))
    # tp_list.append(1.0)
    # fp_list.append(1.0)
    # print tp_list
    area.sort(key=lambda x: x[0])
    tp_list = [x[0] for x in area]
    fp_list = [x[1] for x in area]
    return np.trapz(np.array(tp_list), np.array(fp_list))


def auc_borji(saliency_map, fixation_map, splits=100, stepsize=0.1):
    assert saliency_map.shape == fixation_map.shape

    fixation_map = discretize_gt(fixation_map)
    num_fixations = np.count_nonzero(fixation_map)

    num_pixels = saliency_map.shape[0] * saliency_map.shape[1]

    random_numbers = [
        [np.random.randint(num_pixels) for _ in range(num_fixations)]
        for _ in range(splits)
    ]

    aucs = []
    # for each split, calculate auc
    for i in random_numbers:
        r_sal_map = []
        for k in i:
            r_sal_map.append(saliency_map[k % saliency_map.shape[0] - 1, k / saliency_map.shape[0]])
        # in these values, we need to find thresholds and calculate auc

        thresholds = np.linspace(0, 1, round(1 / stepsize) + 1)[1:-1]
        r_sal_map = np.array(r_sal_map)

        # once threshs are got
        area = []
        area.append((0.0, 0.0))
        for thresh in thresholds:
            # in the saliency map, keep only those pixels with values above threshold
            temp = np.where(saliency_map >= thresh, 1., 0.)

            num_overlap = np.where(np.add(temp, fixation_map) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)

            # fp = (np.sum(temp) - num_overlap)/((np.shape(fixation_map)[0] * np.shape(fixation_map)[1]) - num_fixations)
            # number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
            fp = len(np.where(r_sal_map > thresh)[0]) / (num_fixations * 1.0)

            area.append((round(tp, 4), round(fp, 4)))

        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]

        aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))

    return np.mean(aucs)


def auc_shuff(saliency_map, fixation_map, other_map, splits=100, stepsize=0.1):
    fixation_map = discretize_gt(fixation_map)
    other_map = discretize_gt(other_map)

    num_fixations = np.sum(fixation_map)

    x, y = np.where(other_map == 1)
    other_map_fixs = []
    for j in zip(x, y):
        other_map_fixs.append(j[0] * other_map.shape[0] + j[1])
    ind = len(other_map_fixs)
    assert ind == np.sum(other_map), 'something is wrong in auc shuffle'

    num_fixations_other = min(ind, num_fixations)

    num_pixels = saliency_map.shape[0] * saliency_map.shape[1]
    random_numbers = []
    for i in range(0, splits):
        temp_list = []
        t1 = np.random.permutation(ind)
        for k in t1:
            temp_list.append(other_map_fixs[k])
        random_numbers.append(temp_list)

    aucs = []
    # for each split, calculate auc
    for i in random_numbers:
        r_sal_map = []
        for k in i:
            r_sal_map.append(saliency_map[k % saliency_map.shape[0] - 1, k / saliency_map.shape[0]])
        # in these values, we need to find thresholds and calculate auc
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        r_sal_map = np.array(r_sal_map)

        # once threshs are got
        thresholds = sorted(set(thresholds))
        area = []
        area.append((0.0, 0.0))
        for thresh in thresholds:
            # in the saliency map, keep only those pixels with values above threshold
            temp = np.zeros(saliency_map.shape)
            temp[saliency_map >= thresh] = 1.0
            num_overlap = np.where(np.add(temp, fixation_map) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)

            # fp = (np.sum(temp) - num_overlap)/((np.shape(fixation_map)[0] * np.shape(fixation_map)[1]) - num_fixations)
            # number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
            fp = len(np.where(r_sal_map > thresh)[0]) / (num_fixations * 1.0)

            area.append((round(tp, 4), round(fp, 4)))

        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]

        aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))

    return np.mean(aucs)


def nss(saliency_map, fixation_map):
    f_map_norm = fixation_map / fixation_map.sum()
    s_map_norm = standard_scale(saliency_map)
    return np.sum(s_map_norm * f_map_norm)


def infogain(saliency_map, fixation_map, baseline_map):
    fixation_map = discretize_gt(fixation_map)
    # assuming saliency_map and baseline_map are normalized
    eps = 2.2204e-16

    saliency_map = saliency_map / (np.sum(saliency_map) * 1.0)
    baseline_map = baseline_map / (np.sum(baseline_map) * 1.0)

    # for all places where fixation_map=1, calculate info gain
    temp = []
    x, y = np.where(fixation_map == 1)
    for i in zip(x, y):
        temp.append(np.log2(eps + saliency_map[i[0], i[1]]) - np.log2(eps + baseline_map[i[0], i[1]]))

    return np.mean(temp)


def similarity(saliency_map, fixation_map):
    # here fixation_map is not discretized nor normalized
    saliency_map = min_max_scale(saliency_map)
    fixation_map = min_max_scale(fixation_map)
    saliency_map = saliency_map / (np.sum(saliency_map) * 1.0)
    fixation_map = fixation_map / (np.sum(fixation_map) * 1.0)
    x, y = np.where(fixation_map > 0)
    sim = 0.0
    for i in zip(x, y):
        sim = sim + min(fixation_map[i[0], i[1]], saliency_map[i[0], i[1]])
    return sim


def cc(saliency_map, fixation_map):
    s = standard_scale(saliency_map)
    f = standard_scale(fixation_map)
    r = (s * f).sum() / math.sqrt((s * s).sum() * (f * f).sum())
    return r


def kldiv(saliency_map, fixation_map):
    s_map_sum = saliency_map.sum()
    if not np.allclose(s_map_sum, 0):
        saliency_map = saliency_map / s_map_sum

    f_map_sum = fixation_map.sum()
    if not np.allclose(f_map_sum, 0):
        fixation_map = fixation_map / f_map_sum

    eps = 2.2204e-16
    return np.sum(fixation_map * np.log(eps + fixation_map / (saliency_map + eps)))


if __name__ == '__main__':
    import cv2

    ################################################
    # img = cv2.imread('/home/tarun/mine/tensorflow_examples/image.jpg', 0)
    # print('sim', similarity(img, img))
    # print('cc', cc(img, img))
    # print('kldiv', kldiv(img, img))
    # import sys
    # sys.exit(0)
    #############################################

    # this is just the name its not actually discretized (binary)
    fixation_map = cv2.imread('my_discretised_gt.jpg', 0)

    # d_gt = np.zeros(fixation_map.shape)
    # d_gt[fixation_map>0]=1.0

    saliency_map = cv2.imread('smap_resized.jpg', 0)
    s_map_norm = min_max_scale(saliency_map)

    auc_judd_score = auc_judd(s_map_norm, fixation_map)
    print('auc judd :', auc_judd_score)
    auc_borji_score = auc_borji(s_map_norm, fixation_map)
    print('auc borji :', auc_borji_score)
    auc_shuff_score = auc_shuff(s_map_norm, fixation_map, fixation_map)
    print('auc shuffled :', auc_shuff_score)

    nss_score = nss(saliency_map, fixation_map)
    print('nss :', nss_score)
    infogain_score = infogain(s_map_norm, fixation_map, fixation_map)
    print('info gain :', infogain_score)

    # continous gts
    sim_score = similarity(saliency_map, fixation_map)
    print('sim score :', sim_score)
    cc_score = cc(saliency_map, fixation_map)
    print('cc score :', cc_score)
    kldiv_score = kldiv(saliency_map, fixation_map)
    print('kldiv score :', kldiv_score)
