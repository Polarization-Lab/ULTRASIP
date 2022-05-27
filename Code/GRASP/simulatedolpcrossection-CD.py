from runGRASP import graspRun, pixel
from matplotlib import pyplot as plt
import numpy as np
import datetime as dt



# Path to the YAML file you want to use for the aerosol and surface definition
fwdModelYAMLpath = '/home/cdeleon/ULTRASIP/Code/GRASP/SettingFiles/settings_BiomassBurning.yml'
#fwdModelYAMLpath = '/home/cdeleon/ULTRASIP/Code/GRASP/SettingFiles/settings_1lambda_airmspi_inversion.yml'

#fwdModelYAMLpath = '/home/cdeleon/ULTRASIP/Code/GRASP/SettingFiles/settings_Dust_model1.yml'
#fwdModelYAMLpath = '/home/cdeleon/ULTRASIP/Code/GRASP/SettingFiles/settings_WeaklyAbsorbing1101.yml'
#fwdModelYAMLpath = '/home/cdeleon/ULTRASIP/Code/GRASP/SettingFiles/settings_BiomassBurning2101.yml'
#fwdModelYAMLpath = '/home/cdeleon/ULTRASIP/Code/GRASP/SettingFiles/settings_Dust_model1.yml'
# paths to your GRASP binary and kernels (replace everything up to grasp_open with the path to your GRASP repository)
binPathGRASP = '/home/cdeleon/grasp/build/bin/grasp'
krnlPathGRASP = '/home/cdeleon/grasp/src/retrieval/internal_files'

wvStrFrmt =  '($%4.2f\\mu m$)' # Format of Y-axis labels 
sza = 30 # solar zenith angle
wvls = [0.470] # wavelengths in μm
msTyp = [41, 42, 43] # grasp measurements types (I, Q, U) [must be in ascending order]
azmthΑng = np.r_[0:190:10] # azimuth angles to simulate (0,10,...,175)
vza = 180-np.r_[0:75:5] # viewing zenith angles to simulate [in GRASP cord. sys.]


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

# setup instance of graspRun class, add the above pixel, run grasp and read the output to gr.invRslt[0] (dict)
# releaseYAML=True tmeans that the python will adjust the YAML file to make it correspond to Nwvls (if it does not already)
gr = graspRun(pathYAML=fwdModelYAMLpath, releaseYAML=True, verbose=True)
gr.addPix(nowPix)
gr.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPathGRASP)
#print('AOD at %5.3f μm was %6.4f.' % (gr.invRslt[0]['lambda'][-1],gr.invRslt[0]['aod'][-1]))
#Global Font Size 
plt.rcParams.update({'font.size': 19})
# hemisphere plotting code
Nwvl = len(wvls)

pxInd = 0
r = gr.invRslt[pxInd]['vis'][:,0].reshape(Nazimth, Nvza)
if vza.max()>90: r = 180 - r
theta = gr.invRslt[pxInd]['fis'][:,0].reshape(Nazimth, Nvza)/180*np.pi
r = np.vstack([r, np.flipud(r)]) # mirror symmetric about principle plane
theta = np.vstack([theta, np.flipud(2*np.pi-theta)]) # mirror symmetric about principle plane
#DoLP diff between BB at UV band
for i in range(2):
    data = []
    labels = []
    for l in range(2*Nwvl-1):
        if l<Nwvl:
            wvStr = wvStrFrmt % wvls[l]
            data.append(gr.invRslt[pxInd]['fit_I'][:,l])
           

            if i==1: #1# we need to find DoLP
                Q = gr.invRslt[pxInd]['fit_Q'][:,l]
                U = gr.invRslt[pxInd]['fit_U'][:,l]
                data[-1] = (np.sqrt(Q**2+U**2)/data[-1]) * 100
                

                titlestr='DoLP [%]'
                titlestrdiff='DoLP [%] Difference'
                clrMin = data[-1].min()
                clrMax = data[-1].max()
                print(clrMax)
            #ticksy = np.linspace(clrMin, clrMax,

                labels.append(wvStr)
            # titlestr = 'DoLP'
                name = str(l)
                cmap = 'gist_ncar'
                ticksy = np.linspace(-5,40,5)
            
                data2Ddel = data[-1].reshape(Nazimth, Nvza)
            
                fig2 = plt.figure(figsize=(8, 5))
                slice2Ddel_0deg = data2Ddel[0,:]
                #slice2D_180deg = np.flipud(data2D[int(np.median(np.arange(Nazimth)+1)),:])
                #slice2D = np.hstack([slice2D_180deg,slice2D_0deg]) #<-- FIXED
                #slicetheta = np.linspace(-90,90,2*Nvza)
                slicetheta = np.linspace(0,90,Nvza)
       

                plt.title(wvStr)
                plt.xlabel("Zenith Angle [degrees]")
                plt.yticks(ticksy)
                plt.xticks([0,30,60,90])
                plt.ylabel(titlestr)
                plt.plot(slicetheta,slice2Ddel_0deg)
                
fig2 = plt.figure(figsize=(8, 5))
#plt.title(wvStr)
plt.xlabel("Zenith Angle [°]")
plt.yticks([5,10,15,20,25,30])
plt.xticks([0,30,60,90])
plt.ylabel(titlestr)
plt.minorticks_on()
plt.grid(color='lightgrey', linestyle='-', linewidth=1,which='both')
plt.plot(slicetheta,slice2D_0deg,'b')
plt.plot(slicetheta,slice2Ddel_0deg,'r')
plt.legend(['n=1.50 + 0.01i', 'n=1.50+0.001i'],loc='upper left')# plt.show()
   