import yaml
import numpy as np
import ROOT
import matplotlib.pyplot as plt
import argparse
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from omegaconf import OmegaConf
import math
import parameters
import xray_openBumps
import selection_xray
import os
import mplhep as hep

parser = argparse.ArgumentParser(description='Plot a noise map from a given root file')
parser.add_argument('cfg', help='YAML file with all the analysis parameters', type=str)
args = parser.parse_args()

'''
Parse the default parameters from the default_config.yaml and the custom parameters specified in the command line
'''
base_conf = parameters.get_default_parameters()
second_conf = parameters.get_parameters(args.cfg)
conf = parameters.merge_parameters(base_conf, second_conf)

f_list = []
for file in conf.input.input_file_xray:
    f = conf['input']['input_folder']+file
    f_list.append(f)
f_mask = conf['input']['input_folder']+conf['input']['input_file_mask']
dual   = conf['ph2acf']['is_dual']

out_folder = conf['output']['output_folder']
if not os.path.isdir(out_folder):
    os.makedirs(out_folder)

occupancy_matrix = xray_openBumps.merge_xray_files(conf,filename_list=f_list)
mask_matrix = xray_openBumps.get_pixelalive(conf,f_mask)

if dual == True:
    figsize_min = 13
    figsize_max = 6
    occMap = xray_openBumps.dualMap(conf,occupancy_matrix)
    maskMap = xray_openBumps.dualMap(conf,mask_matrix)
    
else:
    figsize_min = 13
    figsize_max = 10
    occMap = xray_openBumps.quadMap(conf,occupancy_matrix)
    maskMap = xray_openBumps.quadMap(conf,mask_matrix)

maskedMap = selection_xray.mask_map(maskMap)

''' 
Remove the masked pixels from the occ1d plot
'''
occMap_noMasked = selection_xray.remove_mask(occMap, maskedMap)
occ1D_noMasked  = occMap_noMasked.flatten()
openBumpsMap    = selection_xray.open_bumps_xray(conf, occMap, maskedMap)
totalOpenBumps  = np.count_nonzero(openBumpsMap == False)
openBumps_chip1 = np.count_nonzero(openBumpsMap[0:336,0:432] == 0)
openBumps_chip0 = np.count_nonzero(openBumpsMap[0:336,432:864] == 0)
open_bumps_per_chip = [openBumps_chip0,openBumps_chip1]
if dual == False:
    openBumps_chip1 = np.count_nonzero(openBumpsMap[336:672,0:432] == 0)
    openBumps_chip0 = np.count_nonzero(openBumpsMap[336:672,432:864] == 0)
    openBumps_chip3 = np.count_nonzero(openBumpsMap[0:336,432:864] == 0)
    openBumps_chip2 = np.count_nonzero(openBumpsMap[0:336,0:432] == 0)
    open_bumps_per_chip = [openBumps_chip0,openBumps_chip1,openBumps_chip2,openBumps_chip3]
totalMasked     = np.count_nonzero(maskedMap == False)

xray_openBumps.print_results(totalMasked,open_bumps_per_chip)
xray_openBumps.save_cfg(conf)
xray_openBumps.save_csv(conf,totalMasked,open_bumps_per_chip)

print('\n' + 'Producing the plots' + '\n')

##########################################################################################
##########################################################################################
#
#                               FROM HERE ON ONLY PLOTTING
#
##########################################################################################
##########################################################################################

'''
HIT MAP 
2d histogram containing the hit map obtained with the xrays (occupancy*nTriggers)
'''
plt.style.use([hep.cms.style.ROOT])
f0,ax0 = plt.subplots(figsize=(figsize_min,figsize_max))
f0.tight_layout(pad=3)
a0 = ax0.pcolor(occMap, cmap=plt.cm.viridis, vmin=conf['param_xray']['minvcal'],vmax=conf['param_xray']['maxvcal'])
cbar = f0.colorbar(a0, ax=ax0)
cbar.set_label('nHits', labelpad=20)
cbar.formatter.set_powerlimits((0,0))
cbar.formatter.set_useMathText(True)
ax0.set_ylabel('row')
ax0.set_xlabel('column')
f0.savefig(out_folder + 'HitMap_noTitle_' + conf['output']['module_name'], dpi=300)
plt.show()

'''
OCCUPANCY HIST 
histogram of the hits per pixel. The counts around 0 represent the open bumps
'''
f2,ax2 = plt.subplots(figsize=(13,10))
a2 = ax2.hist(occ1D_noMasked, bins=100, range=(float(conf['param_xray']['minvcal']),500), histtype='stepfilled', color = 'orange')
ax2.set_xlabel('nHits')
ax2.set_ylabel('Entries')
ax2.set_yscale('log')
f2.savefig(out_folder + 'Occ1D_noTitle_' + conf['output']['module_name'], dpi=300)
ax2.set_title('Occupancy ' + conf['output']['module_name'], y=1.02)
f2.savefig(out_folder + 'Occ1D_' + conf['output']['module_name'], dpi=300)
plt.show()

'''
OPEN BUMPS
set OpenBumps = 1 and goodBumps = 0
'''
f1,ax1 = plt.subplots(figsize=(figsize_min,figsize_max))
f1.tight_layout(pad=3)
a1 = ax1.pcolor(openBumpsMap, cmap=plt.cm.viridis, vmin=0, vmax=1)
cbar = f1.colorbar(a1, ax=ax1)
ax1.set_ylabel('row')
ax1.set_xlabel('column')
ax1.set_title('Open Bumps Map', y=1.02)
f1.savefig(out_folder + 'OpenBumpsMap_' + conf['output']['module_name'], dpi=300)
plt.show()

print('\n' + 'All plots saved in', out_folder)