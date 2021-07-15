from runGRASP import graspRun, pixel
from matplotlib import pyplot as plt
import numpy as np
import datetime as dt

# Path to the YAML file you want to use for the aerosol and surface definition
fwdModelYAMLpath = '/home/cdeleon/ULTRASIP/Code/GRASP/settings_FWD_IQU_POLAR_2lambda.yml'

# paths to your GRASP binary and kernels (replace everything up to grasp_open with the path to your GRASP repository)
binPathGRASP = '/home/cdeleon/grasp/build/bin/grasp'
krnlPathGRASP = '/home/cdeleon/grasp/src/retrieval/internal_files'

sza = 30 # solar zenith angle
wvls = [0.355, 0.550] # wavelengths in μm
msTyp = [41, 42, 43] # grasp measurements types (I, Q, U) [must be in ascending order]
azmthΑng = np.r_[0:180:10] # azimuth angles to simulate (0,10,...,175)
vza = 180-np.r_[0:75:5] # viewing zenith angles to simulate [in GRASP cord. sys.]
clrMap = plt.cm.jet # color map for plots

# prep geometry inputs for GRASP
Nvza = len(vza)
Nazimth = len(azmthΑng)
thtv_g = np.tile(vza, len(msTyp)*len(azmthΑng)) 
phi_g = np.tile(np.concatenate([np.repeat(φ, len(vza)) for φ in azmthΑng]), len(msTyp))
nbvm = len(thtv_g)/len(msTyp)*np.ones(len(msTyp), int)
meas = np.r_[np.repeat(0.1, nbvm[0]), np.repeat(0.01, nbvm[1]), np.repeat(0.01, nbvm[2])] # dummy values

# define the "pixel" we want to simulate
nowPix = pixel(dt.datetime.now(), 1, 1, 0, 0, masl=0, land_prct=100)
for wvl in wvls: nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv_g, phi_g, meas)
print('made it here')
# setup instance of graspRun class, add the above pixel, run grasp and read the output to gr.invRslt[0] (dict)
# releaseYAML=True tmeans that the python will adjust the YAML file to make it correspond to Nwvls (if it does not already)
gr = graspRun(pathYAML=fwdModelYAMLpath, releaseYAML=True, verbose=True)
gr.addPix(nowPix)
gr.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPathGRASP)
#gr.runGRASP(binPathGRASP=None, krnlPathGRASP=krnlPathGRASP)

# hemisphere plotting code
Nwvl = len(wvls)
print(Nwvl)
fig, ax = plt.subplots(Nwvl, 2, subplot_kw=dict(projection='polar'), figsize=(10,3+3*Nwvl))
if Nwvl == 1: ax = ax[None,:]
pxInd = 0
for l in range(Nwvl):
    r = gr.invRslt[pxInd]['vis'][:,l].reshape(Nazimth, Nvza)
    if vza.max()>90: r = 180 - r
    theta = gr.invRslt[pxInd]['fis'][:,l].reshape(Nazimth, Nvza)/180*np.pi
    r = np.vstack([r, np.flipud(r)]) # mirror symmetric about principle plane
    theta = np.vstack([theta, np.flipud(2*np.pi-theta)]) # mirror symmetric about principle plane
    for i in range(2):
        data = gr.invRslt[pxInd]['fit_I'][:,l]
        # if i==0: data = gr.invRslt[pxInd]['sca_ang'][:,l]
        if i==1:
            Q = gr.invRslt[pxInd]['fit_Q'][:,l]
            U = gr.invRslt[pxInd]['fit_U'][:,l]
            data = np.sqrt(Q**2+U**2)/data
        clrMin = data.min()
        clrMax = data.max()
        v = np.linspace(clrMin, clrMax, 200, endpoint=True)
        ticks = np.linspace(clrMin, clrMax, 3, endpoint=True)
        data2D = data.reshape(Nazimth, Nvza)
        dataFullHemisphere = np.vstack([data2D, np.flipud(data2D)]) # mirror symmetric about principle plane
        c = ax[l,i].contourf(theta, r, dataFullHemisphere, v, cmap=clrMap)
        ax[l,i].plot(np.pi, sza, '.',  color=[1,1,0], markersize=10)
        cb = plt.colorbar(c, orientation='horizontal', ax=ax[l,i], ticks=ticks)
        ax[l,i].set_ylim([0, r.max()])
    wvStr = ' ($%4.2f\\mu m$)' % wvls[l]
    ax[l,0].set_ylabel('I' + wvStr, labelpad=30)
    ax[l,1].set_ylabel('DoLP' + wvStr, labelpad=30)
ttlStr = "BIO-2 From OMI"
plt.suptitle(ttlStr)
plt.tight_layout(rect=[0.01, 0.01,0.98, 0.98])
plt.ion()
plt.show()
