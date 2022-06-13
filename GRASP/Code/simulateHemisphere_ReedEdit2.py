from runGRASP import graspRun, pixel
from matplotlib import pyplot as plt
import numpy as np
import datetime as dt



# Path to the YAML file you want to use for the aerosol and surface definition
fwdModelYAMLpath = '/home/cdeleon/ULTRASIP/Code/GRASP/SettingFiles/settings_BiomassBurning.yml'
#fwdModelYAMLpath = '/home/cdeleon/ULTRASIP/Code/GRASP/SettingFiles/settings_Dust_model1.yml'
#fwdModelYAMLpath = '/home/cdeleon/ULTRASIP/Code/GRASP/SettingFiles/settings_WeaklyAbsorbing1101.yml'
#fwdModelYAMLpath = '/home/cdeleon/ULTRASIP/Code/GRASP/SettingFiles/settings_BiomassBurning2101.yml'
#fwdModelYAMLpath = '/home/cdeleon/ULTRASIP/Code/GRASP/SettingFiles/settings_Dust_model1.yml'
# paths to your GRASP binary and kernels (replace everything up to grasp_open with the path to your GRASP repository)
binPathGRASP = '/home/cdeleon/grasp/build/bin/grasp'
krnlPathGRASP = '/home/cdeleon/grasp/src/retrieval/internal_files'

ttlStr = 'Dust (multi scattering)' # Top Title
wvStrFrmt =  '($%4.2f\\mu m$)' # Format of Y-axis labels 
sza = 30 # solar zenith angle
wvls = [0.55] # wavelengths in μm
msTyp = [41, 42, 43] # grasp measurements types (I, Q, U) [must be in ascending order]
azmthΑng = np.r_[0:190:10] # azimuth angles to simulate (0,10,...,175)
vza = 180-np.r_[0:75:5] # viewing zenith angles to simulate [in GRASP cord. sys.]
clrMapReg = plt.cm.jet # color map for plots
clrMapDiff = plt.cm.seismic # color map for plots

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
# print(Nwvl)
#ax = plt.subplots(2*Nwvl-1, 2, subplot_kw=dict(projection='polar'))#, figsize=(6,6+3*(Nwvl-1)))
#ax = plt.plot(dict(projection='polar'))#, figsize=(6,6+3*(Nwvl-1)))
#fig = plt.figure()
#ax = fig.add_subplot(projection='polar')
#if Nwvl == 1: 
 #   ax = ax[None,:]
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
           

            if i==0: #1# we need to find DoLP
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

                
            if i==10: #0
            #     data[-1] = 1*(np.log10(data[-1]))
            #     titlestr='log(Reflectance)'
            #     titlestrdiff='log(Reflectance) Difference'

            # clrMin = 0#data[-1].min()
            # clrMax = 45#data[-1].max()
            # #ticksy = np.linspace(clrMin, clrMax,5)
            # #else:
            #    # clrMin = data[-1].min(0)
            #    # clrMax = data[-1].max(100)
            # #clrMap = clrMapReg
            # name = str(l)  
            # cmap = 'cool'  
             ticksy = np.linspace(-5, 40,5)
        if i==300:
        #else: # this is a difference plot      
#             if i==1:
#             data.append(data[l-Nwvl+1] - data[0])
#             labels.append(labels[l-Nwvl+1] + " – " + labels[0])
#             titlestr = titlestrdiff
#             wvStr = '(0.55μm (n = 0.01 + i0.01)- 0.34μm (n = 0.01 + i0.001))'
#             name = str(l)+str(i)
#             cmap = 'seismic'
#   #           else:
# #                 data.append(1 - (data[l-Nwvl+1]/data[0]))
# #                 labels.append("1 - " + labels[l-Nwvl+1] + "/" + labels[0])
#             clrMax = np.abs(data[-1]).max()
#             clrMin = -clrMax
              ticksy = np.linspace(clrMin, clrMax,5)

            #clrMap = clrMapDiff
            #fig=plt.figure()
        #unfig = plt.figure()
        
        #unax = plt.subplots(subplot_kw=dict(projection='polar'))#, figsize=(6,6+3*(Nwvl-1)))
       # ax = plt.subplots()#, figsize=(6,6+3*(Nwvl-1)))

        #unv = np.linspace(clrMin, clrMax, 120, endpoint=True)
        #unticks = np.linspace(clrMin, clrMax, 4, endpoint=True)
        #ticks = np.linspace(0, 45, 5)

        data2D = data[-1].reshape(Nazimth, Nvza)
        

        #fixed phi at 0 deg 
        
   #     undataFullHemisphere = np.vstack([data2D, np.flipud(data2D)]) # mirror symmetric about principle plane
    #    unc = plt.contourf(theta, r, dataFullHemisphere, v, cmap=cmap)#cmap='tab20c')
       # c = plt.hist(data[0])#cmap='tab20c')

       # unplt.title(wvStr)
        plt.ylabel(wvStr, labelpad=35)
       # unplt.plot(np.pi, sza, '.',  color=[1,1,0], markersize=25)
       # plt.savefig(f'/home/cdeleon/ULTRASIP/Code/GRASP/Plots/NEW3{name}d1{titlestr}.png')

#ax.set_ylim([0, r.max()])
        #if i==0: 
#ax.set_ylabel(labels[-1], labelpad=80)
    
       #un cb= plt.colorbar(c, orientation='vertical', ticks=ticks,pad=0.1)
        
        plt.gcf()
        fig2 = plt.figure(figsize=(8, 5))
        slice2D_0deg = data2D[0,:]
        slice2D_180deg = np.flipud(data2D[int(np.median(np.arange(Nazimth)+1)),:])
        #slice2D = np.hstack([slice2D_0deg,slice2D_180deg]) <--- OG 
        slice2D = np.hstack([slice2D_180deg,slice2D_0deg]) #<-- FIXED
        slicetheta = np.linspace(-90,90,2*Nvza)
       

        # plt.title(wvStr)
        # plt.xlabel("Zenith Angle [degrees]")
        # plt.yticks(ticksy)
        # plt.xticks([-90,-60,-30,0,30,60,90])
        # #plt.yticks(ticksy)
        # plt.ylabel(titlestr)
        plt.plot(slicetheta,slice2D)
        #plt.show()
    
        #plt.savefig(f'/home/cdeleon/ULTRASIP/Code/GRASP/Plots/NEW2{name}d1{titlestr}.png')
        #plt.savefig(f'/home/cdeleon/ULTRASIP/Code/GRASP/Plots/{cmap}colorbar.png')

#cb = plt.colorbar(c, orientation='horizontal', ax=ax[l,i], ticks=ticks)
#cb = plt.colorbar(c, orientation='horizontal', ax=ax[l,i], ticks=ticks)

#ax[0,0].set_title('log10(Reflectance)')
#ax[0,1].set_title('DoLP [%]')
# plt.suptitle(ttlStr)
#plt.tight_layout(rect=[0.5, 0.5,0.98, 0.98])

#plt.tight_layout(rect=[0.01, 0.01,0.98, 0.98])
#plt.ion()
#plt.show()
#x=range(len(data[2]))
#fig=plt.figure() 
#plt.scatter(x,data[2])     
#plt.hist(data[2],facecolor='blue',edgecolor='red',bins=6)
