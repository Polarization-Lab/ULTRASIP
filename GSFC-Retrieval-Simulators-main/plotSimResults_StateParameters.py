#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will summarize the accuracy of the state parameters retrieved in a simulation through box and whisker plots, among other methods """

import os
import sys
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import copy
import pylab
import itertools
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../GSFC-GRASP-Python-Interface'))
from simulateRetrieval import simulation
import miscFunctions as mf
import ACCP_functions as af

# instruments = ['Lidar090','Lidar090Night','Lidar050Night','Lidar050','Lidar060Night','Lidar060', 'polar07', 'polar07GPM', \
#                 'Lidar090+polar07','Lidar090+polar07GPM','Lidar050+polar07','Lidar060+polar07'] # 7 N=231

instruments = ['polar07-G5NR', 'polar07-DRS']

# casLets = ['a', 'b', 'c', 'd', 'e', 'f','i']
# conCases = ['case06'+caseLet+surf for caseLet in casLets for surf in ['','Desert', 'Vegetation']]
# conCases = ['SPA'+surf for surf in ['','Desert', 'Vegetation']]
conCases = ['case08%c%d' % (let,num) for let in map(chr, range(97, 112)) for num in [1,2]] # a1,a2,b1,..,o2 #30
tauVals = [1.0] 
# tauVals = [0.07,0.08,0.09] 
N = len(conCases)*len(tauVals)
barVals = instruments # each bar will represent on of this category, also need to update definition of N above and the definition of paramTple (~ln75)

trgtλ =  0.532
χthresh= 2 # χ^2 threshold on points
forceχ2Calc = True
minSaved=13

def lgTxtTransform(lgTxt):
    if re.match('.*Coarse[A-z]*Nonsph', lgTxt): # conCase in leg
        return 'Fine[Sphere]\nCoarse[Nonsph]'
    if re.match('.*Fine[A-z]*Nonsph', lgTxt): # conCase in leg
        return 'Fine[Nonsph]\nCoarse[Sphere]'
    if 'misr' in lgTxt.lower(): # instrument in leg
        return lgTxt.replace('Misr','+misr').replace('Polar','+polar').upper()
    if 'lidar' in lgTxt.lower() or 'polar' in lgTxt.lower(): # instrument in leg
        return lgTxt
    return 'Fine[Sphere]\nCoarse[Sphere]' # conCase in leg

    
""" totVars – a list of retrieved variables to include in the plot; options are:
    'ssaMode_fine', 'rEffCalc', 'aodMode_PBL[FT]', 'n_PBL[FT]', 'rEffMode_PBL[FT]', 
    'k_fine', 'n', 'n_fine', 'k', 'k_PBL[FT]', 'rEffMode_fine', 'aod', 'ssa', 
    'ssaMode_PBL[FT]', 'LidarRatio', 'aodMode_fine' """

totVars = np.flipud(['aod', 'ssa', 'n_PBL', 'n_FT', 'n_PBL', 'n_FT', 'rEff', 'LidarRatio'])
totVars = np.flipud(['aod', 'ssa', 'n_PBL', 'n_FT', 'k_PBL', 'k_FT', 'rEffCalc'])
totVars = np.flipud(['aod', 'ssa', 'n_PBL', 'n_FT', 'k_PBL', 'k_FT'])
totVars = np.flipud(['aod', 'ssa', 'n', 'k'])

savePtrnNR = '/Users/wrespino/Synced/Working/SIM_OSSE_Test/ss450-g5nr.leV30.GRASP.YAML*-n*pixStrt*.polar07.random.20060801_0000z.pkl'
savePtrnDRS = '/Users/wrespino/Synced/Working/SIM17_SITA_SeptAssessment/DRS_V01_polar07GPM_case08*_tFct1.00_orbGPM_multiAngles_n*_nAngALL.pkl'


cm = pylab.get_cmap('viridis')

# def getGVlabels(totVars, modVars)
gvNames = copy.copy(totVars)
gvNames = ['$'+mv.replace('Mode','').replace('_fine','_{fine}').replace('_coarse','_{coarse}').replace('rEffCalc','r_{eff}').replace('ssa', 'SSA').replace('aod','AOD').replace('PBL','{PBL}').replace('FT','{FT}')+'$' for mv in gvNames]

Nvars = len(gvNames)
barLeg = []
totBias = dict([]) # only valid for last of whatever we itterate through on the next line
Nbar = len(barVals)
barVals = [y if type(y) is list else [y] for y in barVals] # barVals should be a list of lists
for barInd, barVal in enumerate(barVals):
    savePtrn = savePtrnDRS if 'drs' in barVal[0].lower() else savePtrnNR
    savePaths = glob.glob(savePtrn)
#     savePaths = savePaths[0:5] # HACK
    N = len(savePaths)
    if N<2: assert False, 'Less than two files found for search string %s!' % savePtrn
    harvest = np.zeros([N*len(barVal), Nvars])
    runNames = []    
    for n in range(N*len(barVal)):
        simB = simulation(picklePath=savePaths[n])
        # WE SHOULD DO THIS RIGHT LATER - it is reasonable to expect some back retrievals will fail, need to remove corresponding fwd
        if 'drs' not in barVal[0].lower():
            simB.rsltFwd = simB.rsltFwd[[rf['datetime'] in [rb['datetime'] for rb in simB.rsltBck] for rf in  simB.rsltFwd]]
        NsimsFull = len(simB.rsltBck)
        lInd = np.argmin(np.abs(simB.rsltFwd[0]['lambda']-trgtλ))
        Nsims = len(simB.rsltBck)
        for rf in simB.rsltFwd:
            if 'rEffCalc' not in rf: rf['rEffCalc'] = rf['rEff']
            if 'aodMode' not in rf and 'drs' not in barVal[0].lower(): rf['aodMode'] = rf['aod'][None,:] # OSSE only has one mode
        print("<><><><><><>")
        print(savePaths[n])
        print('AOD=%4.2f, Nsim=%d' % (simB.rsltFwd[0]['aod'][lInd], Nsims))
        print('Spectral variables for λ = %4.2f μm'% simB.rsltFwd[0]['lambda'][lInd])        
        simB.conerganceFilter(χthresh=χthresh, forceχ2Calc=forceχ2Calc, minSaved=minSaved, verbose=True)
        if 'drs' in barVal[0].lower():
            fineIndFwd, fineIndBck = af.findFineModes(simB)
            hghtCut = af.findLayerSeperation(simB.rsltFwd[0], defaultVal=2100)
            strInputs = (','.join(str(x) for x in fineIndFwd), ','.join(str(x) for x in fineIndBck), hghtCut)
            rmse, bias, true = simB.analyzeSim(lInd, fineModesFwd=fineIndFwd, fineModesBck=fineIndBck, hghtCut=hghtCut)
            print('fwd fine mode inds: %s | bck fine mode inds: %s | bot/top layer split: %d m' % strInputs)
        else:
            rmse, bias, true = simB.analyzeSim(lInd, modeCut=0.5)
        qScore, mBias, σScore = af.normalizeError(rmse, bias, true, enhanced=True)
        harvest[n, :] = af.prepHarvest(σScore, totVars) # takes any of qScore, mBias or σScore from normalizeError() above (this is what is plotted)
        print('--------------------')
    plt.rcParams.update({'font.size': 12})
    if barInd==0: 
        figB, axB = plt.subplots(figsize=(4.8,6)) # THIS IS THE BOXPLOT
        yAxMax = Nbar*Nvars-1.3 # UPPER BOUND OF Y-AXIS
        axB.plot([1,1], [-1, yAxMax], ':', color=0.65*np.ones(3)) # vertical line at unity
    pos = Nbar*np.r_[0:harvest.shape[1]]+0.635*barInd-0.4
    hnd = axB.boxplot(harvest, vert=0, patch_artist=True, positions=pos[0:Nvars], sym='.')
    [hnd['boxes'][i].set_facecolor(cm((barInd)/(Nbar))) for i in range(len(hnd['boxes']))]
    [hf.set_markeredgecolor(cm((barInd)/(Nbar))) for hf in hnd['fliers']]
    barLeg.append(hnd['boxes'][0])
axB.set_xscale('log')
axB.set_xlim([0.08,31])
axB.set_ylim([-0.8, yAxMax])
plt.sca(axB)
plt.yticks(Nbar*(np.r_[0:Nvars]+0.1*Nbar-0.2), gvNames)
lgTxt = [lgTxtTransform('%s' % τ) for τ in np.array(barVals)[:,0]]
# lgTxt = ['Perfect Model','Coarse Nonsph','Fine Nonsph', 'Unmodeled WLR','2 extra modes']
lgHnd = axB.legend(barLeg[::-1], ['%s' % τ for τ in lgTxt[::-1]], loc='center left', prop={'size': 9, 'weight':'bold'})
lgHnd.draggable()
axB.yaxis.set_tick_params(length=0)
figB.tight_layout()

plt.ion()
plt.show()

sys.exit()
figSavePath = '/Users/wrespino/Synced/Presentations/ClimateRadiationSeminar_2020/figures/allFiveConditions_misrModis_noFineModeVarsShown_case.png' # % concases[0]
figB.savefig(figSavePath, dpi=600, facecolor='w', edgecolor='w', orientation='portrait', pad_inches=0.05)
figB.savefig(figSavePath[:-3]+'pdf', dpi=600, facecolor='w', edgecolor='w', orientation='portrait', pad_inches=0.05)


