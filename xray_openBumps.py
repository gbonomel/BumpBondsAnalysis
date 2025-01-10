import ROOT
import numpy as np
import matplotlib.pyplot as plt
import parameters
import math
from omegaconf import OmegaConf
import csv
import os

def get_rootpath(hybrid, chip):
    return f'Detector/Board_0/OpticalGroup_0/Hybrid_{hybrid}/Chip_{chip}'

def get_rootprefix(hybrid):
    return f'D_B(0)_O(0)_H({hybrid})'

def TH2F_to_matrix(hist, conf):
    nx, ny = hist.GetNbinsX(), hist.GetNbinsY()
    matrix = np.ndarray((nx, ny))
    for i in range(nx):
        for j in range(ny):
            matrix[i,j] = hist.GetBinContent(i+1, j+1)*(conf['xray_selection']['Ntriggers'])
    return matrix

def TH1F_to_matrix(hist):
    nx = hist.GetNbinsX() + 2
    xdata = np.ndarray(nx) 
    ydata = np.ndarray(nx)
    for i in range(nx):
        xdata[i] = hist.GetXaxis().GetBinCenter(i)
        ydata[i] = hist.GetBinContent(i)
    return np.vstack((xdata, ydata))

# takes the difference between 2d plots and puts in a matrix the new 2d plot
def diff_TH2F_to_matrix(hist1, hist2):
    nx, ny = hist1.GetNbinsX(), hist1.GetNbinsY()
    matrix = np.ndarray((nx, ny))
    for i in range(nx):
        for j in range(ny):
            matrix[i,j] = hist1.GetBinContent(i+1, j+1) - hist2.GetBinContent(i+1, j+1)
    return matrix

def merge_xray_files(conf, filename_list):
    occ2d = []
    hybrid = conf['ph2acf']['hybrid']
    root_scan = conf['scans']['pixelalive']
    for chip in conf.ph2acf.chip_id:
        hist_per_chip  = []
        for f in filename_list:
            path = get_rootpath(hybrid,chip)
            prefix = get_rootprefix(hybrid)
            f = ROOT.TFile.Open(f)
            try: 
                hist_pixelalive = f.Get(f"{path}/{prefix}_" + root_scan + f"({chip})").GetPrimitive(f"{prefix}_" + root_scan + f"({chip})")
                matrix_pixelalive = TH2F_to_matrix(hist_pixelalive, conf)
            except:
                print('Directory ', f"{path}/{prefix}_" + root_scan + f"({chip})", 'does not exist')
                matrix_pixelalive = np.zeros((432,336)).astype('int')
            hist_per_chip.append(matrix_pixelalive)
            f.Close()
        roc = np.sum(hist_per_chip, 0)
        occ2d.append(roc)
    return occ2d # [roc0, roc1, ...]

def get_pixelalive(conf, filename):
    hybrid = conf['ph2acf']['hybrid']
    root_scan = conf['scans']['pixelalive']
    occupancy = []
    prefix = get_rootprefix(hybrid)
    f = ROOT.TFile.Open(filename)
    for chip in conf.ph2acf.chip_id:
        path = get_rootpath(hybrid,chip)
        try:
            hist_occ = f.Get(f"{path}/{prefix}_" + root_scan + f"({chip})").GetPrimitive(f"{prefix}_" + root_scan + f"({chip})")
            matrix_mask = TH2F_to_matrix(hist_occ, conf)
        except:
            print('Directory ', f"{path}/{prefix}_" + root_scan + f"({chip})", 'does not exist -> set to 0')
            matrix_mask = np.zeros((432,336)).astype('int')
            #print(matrix_mask)
        occupancy.append(matrix_mask)
    f.Close()
    return occupancy

def quadMap(conf, occupancy_matrix):
    row  = conf['ph2acf']['rows']
    col  = conf['ph2acf']['columns']
    occupancyMap = []
    for chip in conf.ph2acf.chip_id:
        if chip in range(0,2):
            occupancy_matrix[chip] = np.rot90(occupancy_matrix[chip])
            occupancy_matrix[chip] = np.flip(occupancy_matrix[chip],1)
            occupancy_matrix[chip] = np.rot90(occupancy_matrix[chip])
        else:
            #print(occupancy_matrix[chip])
            occupancy_matrix[chip] = np.flip(occupancy_matrix[chip],0)
        occupancyMap.append(occupancy_matrix[chip]) 
    hitmap = np.zeros((864,672)).astype('int')
    hitmap [432:864, 0:336]   = occupancyMap[0]
    hitmap [0:432, 0:336]     = occupancyMap[1]
    hitmap [0:432, 336:672]   = occupancyMap[2]
    hitmap [432:864, 336:672] = occupancyMap[3]
    hitmap = np.rot90(hitmap)
    hitmap = np.flip(hitmap,0)
    return hitmap
    
def dualMap(conf, occupancy_matrix):
    row  = conf['ph2acf']['rows']
    col  = conf['ph2acf']['columns']
    hitmap = np.zeros((864,336)).astype('int')
    hitmap [432:864, 0:336] = occupancy_matrix[0]
    hitmap [0:432, 0:336]   = occupancy_matrix[1]
    hitmap = np.rot90(hitmap)
    return hitmap

def print_results(mask,open_bumps): #open_bumps = [chip0, chip1, chip2, chip3]
    total_open_bumps = sum(open_bumps)
    err = math.sqrt(total_open_bumps)
    print('Masked pixels: ', mask)
    print('Total open bumps detected: ', total_open_bumps, ' +- {0:1.1f}'.format(err))
    for i in range(len(open_bumps)):    
        print(f'chip{i}: ', open_bumps[i])

def save_cfg(conf):
    out_folder = conf['output']['output_folder']+ 'config/'
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    out_yaml = out_folder+conf['output']['module_name']+".yaml"
    OmegaConf.save(config=conf, f=out_yaml)
    print('\n'+ 'Configuration file saved in', out_yaml + '\n')

def save_csv(conf,mask,open_bumps_per_chip): #open_bumps_per_chip = [chip0, chip1, chip2, chip3]
    out_folder = conf['output']['output_folder']
    dual = conf['ph2acf']['is_dual']
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    total_open_bumps = sum(open_bumps_per_chip)
    err = math.sqrt(total_open_bumps)
    chips  = ['chip0','chip1']
    header = ['cycle','masked','total_bumps','err']
    if dual == False:
        chips.append('chip2')
        chips.append('chip3')
    data   = [conf['ph2acf']['thermal_cycles'],mask,total_open_bumps,'{0:1.1f}'.format(err)]
    for i in range(len(open_bumps_per_chip)):
        header.append(chips[i])
        data.append(open_bumps_per_chip[i])
    filename = out_folder + 'openBumps_xray_' + conf['output']['module_name'] + '.csv'
    with open(filename, 'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(data)
    print('Results saved in', filename)    

