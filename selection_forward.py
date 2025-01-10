import ROOT
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
import parameters

# consider bad bumps those that give fit errors at -80V and good bumps those that have fit errors at +1 only
def fit_errors_selection(fiterr_pos, fiterr_neg, matrix):
    negative_bias_selection = np.where((fiterr_neg == 1), 0.0, matrix)
    fitErr_selection = np.where((fiterr_pos == 1), np.nan, negative_bias_selection) #put it to a None 
    return fitErr_selection

#defines a matrix with the masked pixels
def mask_map(threshold2d):
    mask = np.where((threshold2d == 0), False, True)
    return mask

def remove_mask(matrix, bool_mask_matrix):
    maskFilter = np.where((bool_mask_matrix==False), np.nan, matrix)
    return maskFilter

#remove the masked pixels from the analysis by setting the matrix value to a random high number
def mask_selection(mask_matrix, output_matrix):
    newMatrix = np.where((mask_matrix == 0), None, output_matrix)
    return newMatrix

def open_bumps_forward_bias(conf, noise2d, noise_diff, threshold2d, thr_diff):
    cut_noise = conf['forward_bias_selection']['noise_cut']
    cut_thr   = conf['forward_bias_selection']['thr_cut']
    openMatrix = np.where(((abs(noise_diff) < cut_noise) & (abs(thr_diff) < cut_thr)), 0.0, 1)
    mask_filer_openMap = np.where(((threshold2d == 0) | (noise2d == 0)), np.nan, openMatrix)
    return mask_filer_openMap
