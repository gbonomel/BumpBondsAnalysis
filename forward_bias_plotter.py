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
import forward_openBumps
import selection_forward
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
#print(OmegaConf.to_yaml(conf))

'''
Some plotting parameters from the config
'''
module        = conf['output']['module_name']
maxvcal_thr   = conf['param_forward_bias']['maxVcal_thr']
minvcal_thr   = conf['param_forward_bias']['minVcal_thr']
maxvcal_noise = conf['param_forward_bias']['maxVcal_noise']
minvcal_noise = conf['param_forward_bias']['minVcal_noise']
outfolder     = conf['output']['output_folder']
dual          = conf['ph2acf']['is_dual']
if not os.path.isdir(outfolder):
    os.makedirs(outfolder)

thr2d_pos   = []
thr2d_neg   = []
noise2d_pos = []
noise2d_neg = []

thrMap_diff = []
noiseMap_diff = []

thr2d_pos   = forward_openBumps.get_thr2d_pos(conf)
thr2d_neg   = forward_openBumps.get_thr2d_neg(conf)
noise2d_pos = forward_openBumps.get_noise2d_pos(conf)
noise2d_neg = forward_openBumps.get_noise2d_neg(conf)
fiterr_pos_list  = forward_openBumps.get_fiterrors_pos(conf)
fiterr_neg_list  = forward_openBumps.get_fiterrors_neg(conf)

for chip_id in range(len(fiterr_pos_list)):
    '''
    take the difference of 2D thr and noise distribution for the negative and forward bias into matrix and save them in the respective lists (list number == chipID)
    '''
    thr2d_diff   = thr2d_pos[chip_id] - thr2d_neg[chip_id] 
    noise2d_diff = noise2d_pos[chip_id] - noise2d_neg[chip_id]
    thr2d_diff   = selection_forward.fit_errors_selection(fiterr_pos_list[chip_id], fiterr_neg_list[chip_id],thr2d_diff)
    noise2d_diff = selection_forward.fit_errors_selection(fiterr_pos_list[chip_id], fiterr_neg_list[chip_id],noise2d_diff)
    thrMap_diff.append(thr2d_diff)
    noiseMap_diff.append(noise2d_diff)

#print(thr2d_neg)

if dual == True:
    figsize_min = 13
    figsize_max = 6
    thrMap_neg   = forward_openBumps.dualMap(conf, thr2d_neg)
    thrMap_pos   = forward_openBumps.dualMap(conf, thr2d_pos)
    noiseMap_neg = forward_openBumps.dualMap(conf, noise2d_neg)
    noiseMap_pos = forward_openBumps.dualMap(conf, noise2d_pos)

    thrDiff      = forward_openBumps.dualMap(conf, thrMap_diff)
    noiseDiff    = forward_openBumps.dualMap(conf, noiseMap_diff)

    mask      = selection_forward.mask_map(thrMap_neg)
    thrDiff   = selection_forward.remove_mask(thrDiff,mask)
    noiseDiff = selection_forward.remove_mask(noiseDiff,mask)

else:
    figsize_min = 13
    figsize_max = 10
    thrMap_neg   = forward_openBumps.quadMap(conf, thr2d_neg)
    thrMap_pos   = forward_openBumps.quadMap(conf, thr2d_pos)
    noiseMap_neg = forward_openBumps.quadMap(conf, noise2d_neg)
    noiseMap_pos = forward_openBumps.quadMap(conf, noise2d_pos)

    thrDiff      = forward_openBumps.quadMap(conf, thrMap_diff)
    noiseDiff    = forward_openBumps.quadMap(conf, noiseMap_diff)

    mask      = selection_forward.mask_map(thrMap_neg)
    thrDiff   = selection_forward.remove_mask(thrDiff,mask)
    noiseDiff = selection_forward.remove_mask(noiseDiff,mask)

'''
Open bumps map with masked pixels put to np.nan
'''
openBumpsMap    = selection_forward.open_bumps_forward_bias(conf, noiseMap_neg, noiseDiff, thrMap_neg, thrDiff)
totalOpenBumps  = np.count_nonzero(openBumpsMap == 0)
openBumps_chip0 = np.count_nonzero(openBumpsMap[0:336,432:864] == 0)
openBumps_chip1 = np.count_nonzero(openBumpsMap[0:336,0:432] == 0)
open_bumps_per_chip = [openBumps_chip0,openBumps_chip1]
if dual == False:
    openBumps_chip1 = np.count_nonzero(openBumpsMap[336:672,0:432] == 0)
    openBumps_chip0 = np.count_nonzero(openBumpsMap[336:672,432:864] == 0)
    openBumps_chip3 = np.count_nonzero(openBumpsMap[0:336,432:864] == 0)
    openBumps_chip2 = np.count_nonzero(openBumpsMap[0:336,0:432] == 0)
    open_bumps_per_chip = [openBumps_chip0,openBumps_chip1,openBumps_chip2,openBumps_chip3]
totalMasked     = np.count_nonzero(mask == 0)


forward_openBumps.print_results(totalMasked,open_bumps_per_chip)
forward_openBumps.save_cfg(conf)
forward_openBumps.save_csv(conf,totalMasked,open_bumps_per_chip)

print('\n' + 'Producing the plots' + '\n')



##########################################################################################
##########################################################################################
#
#                               FROM HERE ON ONLY PLOTTING
#
##########################################################################################
##########################################################################################


'''
OPEN BUMPS
set OpenBumps = 1 and goodBumps = 0
'''
plt.style.use([hep.cms.style.ROOT])
f0,ax0 = plt.subplots(figsize=(figsize_min,figsize_max))
f0.tight_layout(pad=3)
a0 = ax0.pcolor(openBumpsMap, cmap=plt.cm.viridis, vmin=0, vmax=1)
cbar = f0.colorbar(a0, ax=ax0)
ax0.set_ylabel('row')
ax0.set_xlabel('column')
f0.savefig(outfolder + 'OpenBumpsMap_noTitle_' + module, dpi=300)
ax0.set_title('Open Bumps Map', y=1.02)
f0.savefig(outfolder + 'OpenBumpsMap_' + module, dpi=300)
plt.show()

'''
MASKED APPLIED
set masked = 0.
'''
f2,ax2 = plt.subplots(figsize=(figsize_min,figsize_max))
f2.tight_layout(pad=3)
a2 = ax2.pcolor(mask, cmap=plt.cm.viridis, vmin=0, vmax=1)
cbar = f2.colorbar(a2, ax=ax2)
ax2.set_ylabel('row')
ax2.set_xlabel('column')
f2.savefig(outfolder + 'MaskedPixels_noTitle_' + module, dpi=300)
ax2.set_title('Masked Pixels Map', y=1.02)
f2.savefig(outfolder + 'MaskedPixels_' + module, dpi=300)
plt.show()

'''
THRESHOLD DIFFERENCE HIST 
1d histogram containing the difference between the forward (+1V) and negative (-80V) bias threshold histograms 
'''
f3,ax3 = plt.subplots(figsize=(13,10))
a3 = ax3.hist(thrDiff.flatten(), bins=100, range=(minvcal_thr,maxvcal_thr), histtype='stepfilled', color = 'orange')
ax3.set_xlabel('Threshold [$\Delta$VCAL]')
ax3.set_ylabel('Entries')
ax3.set_xlim([minvcal_thr, maxvcal_thr])
#ax3.set_yscale('log')
f3.savefig(outfolder + '1DThresholdDiff_noTitle_' + module, dpi=300)
ax3.set_title('Threshold Difference Histogram ' + module, y=1.02)
f3.savefig(outfolder + '1DThresholdDiff_' + module, dpi=300)
plt.show()

'''
NOISE DIFFERENCE HIST 
1d histogram containing the difference between the forward (+1V) and negative (-80V) bias noise histograms 
'''
f4,ax4 = plt.subplots(figsize=(13,10))
a4 = ax4.hist(noiseDiff.flatten(), bins=30, range=(minvcal_noise,maxvcal_noise), histtype='stepfilled')
ax4.set_xlabel('Noise [$\Delta$ENC]')
ax4.set_ylabel('Entries')
ax4.set_xlim([minvcal_noise, maxvcal_noise])
#ax4.set_yscale('log')
f4.savefig(outfolder + '1DNoiseDiff_noTitle_' + module, dpi=300)
ax4.set_title('Noise Difference Histogram ' + module, y=1.02)
f4.savefig(outfolder + '1DNoiseDiff_' + module, dpi=300)
plt.show()

'''
THRESHOLD DIFFERENCE MAP 
2d histogram containing the difference between the forward (+1V) and negative (-80V) bias threshold distributions 
'''
f5,ax5 = plt.subplots(figsize=(figsize_min,figsize_max))
f5.tight_layout(pad=3)
a5 = ax5.pcolor(thrDiff, cmap=plt.cm.viridis, vmin=0,vmax=maxvcal_thr)
cbar = f5.colorbar(a5, ax=ax5)
ax5.set_ylabel('row')
ax5.set_xlabel('column')
f5.savefig(outfolder + '2DThresholdDiff_noTitle_' + module, dpi=300)
ax5.set_title('Threshold Difference Map', y=1.02)
f5.savefig(outfolder + '2DThresholdDiff_' + module, dpi=300)
plt.show()

'''
NOISE DIFFERENCE MAP 
2d histogram containing the difference between the forward (+1V) and negative (-80V) bias noise distributions 
'''
f6,ax6 = plt.subplots(figsize=(figsize_min,figsize_max))
f6.tight_layout(pad=3)
a6 = ax6.pcolor(noiseDiff, cmap=plt.cm.viridis, vmin=0,vmax=maxvcal_noise)
cbar = f6.colorbar(a6, ax=ax6)
ax6.set_ylabel('row')
ax6.set_xlabel('column')
f6.savefig(outfolder + '2DNoiseDiff_noTitle_' + module, dpi=300)
ax6.set_title('Noise Difference Map', y=1.02)
f6.savefig(outfolder + '2DNoiseDiff_' + module, dpi=300)
plt.show()

'''
DELTA NOISE TO DELTA THRESHOLD DISTRIBUTION
2d histogram of the delta noise to delta threshold distribution between the forward (+1V) and the negative (-80V) bias
'''
f7,ax7 = plt.subplots(figsize=(13,10))
a7 = ax7.hist2d(thrDiff.flatten(), noiseDiff.flatten(),  bins = (100,100), cmap=plt.cm.nipy_spectral, range = [[-1200,1200],[-200,500]],norm=colors.LogNorm())
f7.colorbar(a7[3],ax=ax7)
ax7.set_xlabel('$\Delta$ Threshold [VCAL]')
ax7.set_ylabel('$\Delta$ Noise [VCAL]')
#ax7.set_xlim([minvcal_noise, maxvcal_noise])
#ax7.set_xlim([minvcal_thr, maxvcal_thr])
f7.savefig(outfolder + '2DdeltaNoiseToDeltaThr_' + module, dpi=300)
plt.show()

#'''
#DELTA NOISE TO DELTA THRESHOLD DISTRIBUTION
#2d histogram of the delta noise to delta threshold distribution between the forward (+1V) and the negative (-80V) bias
#'''
##fig, ax = plt.subplots(1, 1, figsize=[5,5])
#fig, ax = plt.subplots()
#hist = ax.hist2d(thrDiff.flatten(), noiseDiff.flatten(), bins = (100,100), cmap=plt.cm.nipy_spectral, range = [[-150,200],[-50,maxvcal_noise]],norm=colors.LogNorm())
#x = np.linspace(-conf['forward_bias_selection']['thr_cut'], conf['forward_bias_selection']['thr_cut'], (2*conf['forward_bias_selection']['thr_cut'] + 1))
#ax.fill_between(x, -conf['forward_bias_selection']['noise_cut'], conf['forward_bias_selection']['noise_cut'], edgecolor='red', facecolor='None')
#plt.xlabel('$\Delta$ Threshold [VCAL]')
#plt.ylabel('$\Delta$ Noise [VCAL]')
#cbar = plt.colorbar()
#plt.show()
##plt.title('Noise Difference Map')
#fig.savefig(outfolder + '2DdeltaNoiseToDeltaThr_' + module, dpi=300)

#'''
#NOISE HIST 
#1d histogram containing the forward (+1V) and negative (-80V) bias noise histograms 
#'''
#noise80, bins_80 = np.histogram(noiseMap_neg.flatten(), bins=(maxvcal_noise - minvcal_noise))
#noise1, bins_1   = np.histogram(noiseMap_pos.flatten(), bins=(maxvcal_noise - minvcal_noise))
#plt.stairs(noise80, bins_80, label='Noise1D Negative bias')
#plt.stairs(noise1, bins_1, label='Noise1D Forward bias')
#plt.yscale('log')
#plt.title('Noise Histogram ' + module)
#plt.xlabel('Noise [$\Delta$ENC]')
#plt.ylabel('Entries')
#plt.xlim(0, 400)
#plt.legend()
#plt.savefig(outfolder + '1DNoiseHist_' + module, dpi=300)
#plt.show()
#
#'''
#THRESHOLD HIST 
#1d histogram containing the forward (+1V) and negative (-80V) bias threshold histograms 
#'''
#thr80, bins_80 = np.histogram(thrMap_neg.flatten(), bins=(maxvcal_thr - minvcal_thr))
#thr1, bins_1   = np.histogram(thrMap_pos.flatten(), bins=(maxvcal_thr - minvcal_thr))
#plt.stairs(thr80, bins_80, label='Threshold1D Negative bias')
#plt.stairs(thr1, bins_1, label='Threshold1D Forward bias')
#plt.yscale('log')
#plt.title('Threshold Histogram ' + module)
#plt.xlabel('Threshold [VCAL]')
#plt.ylabel('Entries')
#plt.xlim(0, 2000)
#plt.legend()
#plt.savefig(outfolder + '1DThresholdHist_' + module, dpi=300)
#plt.show()

print('\n' + 'All plots saved in', conf['output']['output_folder'])