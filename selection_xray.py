import ROOT
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
import parameters


def mask_map(mask_matrix):
    newMatrix = np.where((mask_matrix == 0), False, True)
    return newMatrix

def remove_mask(occ_matrix, bool_mask_matrix):
    maskFilter = np.where(~bool_mask_matrix, np.nan, occ_matrix)
    return maskFilter

def open_bumps_xray(conf, occ_matrix, bool_mask_matrix):
    cut_occ = conf['xray_selection']['occupancy_cut']
    openMatrix_selection = np.where((occ_matrix < cut_occ), False, True)
    removeMask = np.where(~bool_mask_matrix, True, openMatrix_selection)
    return removeMask

def shorted_bumps(mask_matrix):
    newMatrix = np.where((mask_matrix > 1.5), False, True)
    return newMatrix