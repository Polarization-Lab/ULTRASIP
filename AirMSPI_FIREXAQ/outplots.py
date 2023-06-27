# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:25:49 2023

@author: ULTRASIP_1
"""

# Import Libraries 
import numpy as np
import matplotlib.pyplot as plt



# Open the text file in read mode
# Define GRASP output file path 
#outpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/June2523/Washington1/Merd_Inchelium.txt"
outpath =  "C:/Users/Clarissa/Documents/GitHub/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/June2523/Washington1/Merd_Inchelium.txt"


file = open(outpath)
content = file.readlines()
n = -1
m=t=n

#Set number of measurements and number of radiometric and polarimetric channels
meas_num = 9
rad_chnls = 7
pol_chnls = 3
# Initialize lists for each column
sza =  [[] for _ in range(rad_chnls)]
scat =  [[] for _ in range(rad_chnls)]

meas_I = [[] for _ in range(rad_chnls)]
fit_I = [[] for _ in range(rad_chnls)]

meas_Q = [[] for _ in range(pol_chnls)]
fit_Q = [[] for _ in range(pol_chnls)]

meas_U = [[] for _ in range(pol_chnls)]
fit_U = [[] for _ in range(pol_chnls)]

for i in range(len(content)):

    if 'meas_I' in content[i]:
        n=n+1
        for num in range(i+1, i+meas_num+1):
            meas_I[n].append(float(content[num].split()[5]))
            fit_I[n].append(float(content[num].split()[6]))
        
            sza[n].append(float(content[num].split()[1]))
            scat[n].append(float(content[num].split()[4]))
    
    elif 'meas_U/I' in content[i]:
        m=m+1
        for num in range(i+1, i+meas_num+1):
            meas_U[m].append(float(content[num].split()[5]))
            fit_U[m].append(float(content[num].split()[6]))
            
    
    elif 'meas_Q/I' in content[i]:
        t=t+1
        for num in range(i+1, i+meas_num+1):
            meas_Q[t].append(float(content[num].split()[5]))
            fit_Q[t].append(float(content[num].split()[6]))
            
            
#Plots 

plt.figure()
plt.plot(scat[0],meas_I[0],linestyle='dashed',color='gray')
plt.plot(scat[0],fit_I[0],color='gray')

plt.plot(scat[1],meas_I[1],linestyle='dashed',color='pink')
plt.plot(scat[1],fit_I[1],color='pink')

plt.plot(scat[2],meas_I[2],linestyle='dashed',color='violet')
plt.plot(scat[2],fit_I[2],color='violet')

plt.plot(scat[3],meas_I[3],linestyle='dashed',color='blue')
plt.plot(scat[3],fit_I[3],color='blue')

plt.plot(scat[4],meas_I[4],linestyle='dashed',color='green')
plt.plot(scat[4],fit_I[4],color='green')

plt.plot(scat[5],meas_I[5],linestyle='dashed',color='red')
plt.plot(scat[5],fit_I[5],color='red')

plt.plot(scat[6],meas_I[6],linestyle='dashed',color='orange')
plt.plot(scat[6],fit_I[6],color='orange')

plt.xlabel('Scattering Angle')
plt.ylabel('BRF(I)')
plt.show()

plt.figure()
plt.plot(scat[3],meas_Q[0],linestyle='dashed',color='blue')
plt.plot(scat[3],fit_Q[0],color='blue')

plt.plot(scat[5],meas_Q[1],linestyle='dashed',color='red')
plt.plot(scat[5],fit_Q[1],color='red')

plt.plot(scat[6],meas_Q[2],linestyle='dashed',color='orange')
plt.plot(scat[6],fit_Q[2],color='orange')

plt.xlabel('Scattering Angle')
plt.ylabel('BRF(Q)')
plt.show()


plt.figure()
plt.plot(scat[3],meas_U[0],linestyle='dashed',color='blue')
plt.plot(scat[3],fit_U[0],color='blue')

plt.plot(scat[5],meas_U[1],linestyle='dashed',color='red')
plt.plot(scat[5],fit_U[1],color='red')

plt.plot(scat[6],meas_U[2],linestyle='dashed',color='orange')
plt.plot(scat[6],fit_U[2],color='orange')

plt.xlabel('Scattering Angle')
plt.ylabel('BRF(U)')
plt.show()

# fig, (ax1,ax2,ax3) = plt.subplots(
#           nrows=3, ncols=1, dpi=240,figsize=(6, 8))
    
    # ax1.scatter(scat[:,0],i_obs[:,0],marker='x',color="green",label="355nm")
    # ax1.scatter(scat[:,1],i_obs[:,1],marker='x',color="pink",label="380nm")
    # ax1.scatter(scat[:,2],i_obs[:,2],marker='x',color="brown",label="445nm")  
    # ax1.scatter(scat[:,4],i_obs[:,4],marker='x',color="purple",label="555nm")
    
# ax1.plot(scat[0],meas_I[0],linestyle='dashed',color="green",label="355nm")
# ax1.plot(scat[1],meas_I[1],linestyle='dashed',color="pink",label="380nm")
# ax1.plot(scat[2],meas_I[2],linestyle='dashed',color="brown",label="445nm")  
# ax1.plot(scat[3],meas_I[3],linestyle='dashed',color="purple",label="555nm")
    

# ax1.plot(scat[0],fit_I[0],color="green")
# ax1.plot(scat[1],fit_I[1],color="pink")
# ax1.plot(scat[2],fit_I[2],color="brown")
# ax1.plot(scat[3],fit_I[3],color="purple")

    # ax1.scatter(scat[:,3],i_obs[:,3],marker='x',color="blue",label="470nm")
    # ax1.scatter(scat[:,5],i_obs[:,5],marker='x',color="red",label="660nm")
    # ax1.scatter(scat[:,6],i_obs[:,6],marker='x',color="orange",label="865nm")
    
# ax1.plot(scat[:,3],i_obs[:,3],linestyle='dashed',color="blue",label="470nm")
# ax1.plot(scat[:,5],i_obs[:,5],linestyle='dashed',color="red",label="660nm")
# ax1.plot(scat[:,6],i_obs[:,6],linestyle='dashed',color="orange",label="865nm")

# ax1.plot(scat[:,3],i_mod[:,3],color="blue")
# ax1.plot(scat[:,5],i_mod[:,5],color="red")
# ax1.plot(scat[:,6],i_mod[:,6],color="orange")
    

    # ax1.get_yaxis().get_major_formatter().labelOnlyBase = True
    # ax1.set_yscale("log")
    # # ax1.set_ylim(5e-2,2.5e-1)
    # # ax1.set_yticks(np.arange(0.03,0.3,0.06))
    # # ax1.set_ylabel('BRF(I)', fontsize='xx-large')
    # ax1.yaxis.set_label_coords(-.15, .5)
    # ax1.xaxis.set_label_coords(.5, -.1)
    # ax1.set_xticks([])
    
    # ax1.legend(loc='best',ncol=4,fontsize='medium')  # Upper right
    # ax1.legend(loc='upper center', bbox_to_anchor=(0.5,3),
    #       ncol=3, fancybox=True, shadow=True,fontsize='large')
    
    # ax2.plot(scat[:,3],q_obs[:,0],linestyle='dashed',color="blue",label="470nm")
    # ax2.plot(scat[:,5],q_obs[:,1],linestyle='dashed',color="red",label="660nm")
    # ax2.plot(scat[:,6],q_obs[:,2],linestyle='dashed',color="orange",label="865nm")
    
    # ax2.plot(scat[:,3],q_mod[:,0],color="blue")
    # ax2.plot(scat[:,5],q_mod[:,1],color="red")
    # ax2.plot(scat[:,6],q_mod[:,2],color="orange")
         
    # ymin = 0.0;
    # ymax = 0.5;
    # yticks = 0.1;
     
    # ax2.set_ylim(ymin,ymax)
    # ax2.set_yticks(np.arange(ymin,ymax,yticks))
    # # ax2.set_ylabel('BRF(Q)', fontsize='xx-large')
    # ax2.yaxis.set_label_coords(-.15, .5)
    # ax2.xaxis.set_label_coords(.5, -.1)
    # ax2.set_xticks([])
    
    # # ax3.scatter(scat[:,3],u_obs[:,0],marker='x',color="blue",label="470nm")
    # # ax3.scatter(scat[:,5],u_obs[:,1],marker='x',color="red",label="660nm")
    # # ax3.scatter(scat[:,6],u_obs[:,2],marker='x',color="orange",label="865nm")
    
    # ax3.plot(scat[:,3],u_obs[:,0],linestyle='dashed',color="blue",label="470nm")
    # ax3.plot(scat[:,5],u_obs[:,1],linestyle='dashed',color="red",label="660nm")
    # ax3.plot(scat[:,6],u_obs[:,2],linestyle='dashed',color="orange",label="865nm")
    
    # ax3.plot(scat[:,3],u_mod[:,0],color="blue")
    # ax3.plot(scat[:,5],u_mod[:,1],color="red")
    # ax3.plot(scat[:,6],u_mod[:,2],color="orange")
         
    # ymin = -0.06;
    # ymax =0.01;
    # yticks = 0.02;
     
    # ax3.set_ylim(ymin,ymax)
    # ax3.set_yticks(np.arange(ymin,ymax,yticks))
    # # ax3.set_ylabel('BRF(U)', fontsize='xx-large')
    
    # # ax3.set_xlim(60,145)
    # # ax3.set_xticks(np.arange(60,145,10))
    # ax3.set_xlim(65,150)
    # ax3.set_xticks(np.arange(65,150,10))
    # ax3.set_xlabel( u"\u03A9 [\u00b0]",fontsize='xx-large')
    
    # ax3.yaxis.set_label_coords(-.15, .5)
    # ax3.xaxis.set_label_coords(.5, -.1)
   
    # plt.tight_layout()  
    # plt.show() 
    
    # fig, ax = plt.subplots(3, 3,
    #                        sharex=True, 
    #                        sharey='row')

    # ax[0,0].plot(scat[:,0],i_obs[:,0],linestyle='dashed',color="green")
    # ax[0,0].plot(scat[:,1],i_obs[:,1],linestyle='dashed',color="pink")
    # ax[0,0].plot(scat[:,2],i_obs[:,2],linestyle='dashed',color="brown")  
    # ax[0,0].plot(scat[:,4],i_obs[:,4],linestyle='dashed',color="purple")
    
    # ax[0,0].set_ylabel('BRF(I)', fontsize='x-large')
    # ax[0,0].yaxis.set_label_coords(-.4, .5)
    
    # ax[0,0].plot(scat[:,0],i_mod[:,0],color="green",label="355")
    # ax[0,0].plot(scat[:,1],i_mod[:,1],color="pink",label="380")
    # ax[0,0].plot(scat[:,2],i_mod[:,2],color="brown",label="445")
    # ax[0,0].plot(scat[:,4],i_mod[:,4],color="purple",label="555")
    
    # ax[0,0].set_yticks([0.06,0.1,0.2])
    
    # ax[0,1].plot(scat[:,3],i_obs[:,3],linestyle='dashed',color="blue")

    # ax[0,1].plot(scat[:,5],i_obs[:,5],linestyle='dashed',color="red")
    # ax[0,1].plot(scat[:,6],i_obs[:,6],linestyle='dashed',color="orange")

    # ax[0,1].plot(scat[:,3],i_mod[:,3],color="blue",label='470')
    # ax[0,1].plot(scat[:,5],i_mod[:,5],color="red",label='660')
    # ax[0,1].plot(scat[:,6],i_mod[:,6],color="orange",label='865')
    
    # ax[0,0].legend(bbox_to_anchor=(3.62,1),
    #       ncol=2,fontsize='medium',frameon=False)
    # ax[0,1].legend(bbox_to_anchor=(1.08,0.5),
    #       ncol=2,fontsize='medium',frameon=False)
    
    # fig.delaxes(ax[0,2])
    
    # ax[2,0].set_ylabel('BRF(U)', fontsize='x-large')
    # ax[2,0].yaxis.set_label_coords(-.4, .5)
    
    # ax[2,0].plot(scat[:,3],u_obs[:,0],linestyle='dashed',color="blue",label="470nm")
    # ax[2,0].fill_between(np.sort(scat[:,3]), u_obs[:,0]-np.std(u_obs[:,0]), u_obs[:,0]+np.std(u_obs[:,0]),color="blue",alpha=0.2)

    # ax[2,1].plot(scat[:,5],u_obs[:,1],linestyle='dashed',color="red",label="660nm")
    # ax[2,1].fill_between(scat[:,5], u_obs[:,1]-np.std(u_obs[:,1]), u_obs[:,1]+np.std(u_obs[:,1]),color="red",alpha=0.2)
    # print(np.std(u_obs[:,1]))
    # ax[2,2].plot(scat[:,6],u_obs[:,2],linestyle='dashed',color="orange",label="865nm")
    # ax[2,2].fill_between(scat[:,6], u_obs[:,2]-np.std(u_obs[:,2]), u_obs[:,2]+np.std(u_obs[:,2]),color="orange",alpha=0.2)

    # ax[2,0].plot(scat[:,3],u_mod[:,0],color="blue")
    # ax[2,1].plot(scat[:,5],u_mod[:,1],color="red")
    # ax[2,2].plot(scat[:,6],u_mod[:,2],color="orange")
    
    # ax[1,0].set_ylabel('BRF(Q)', fontsize='x-large')
    # ax[1,0].yaxis.set_label_coords(-.4, .5)
    
    # ax[1,0].plot(scat[:,3],q_obs[:,0],linestyle='dashed',color="blue",label="470nm")
    # ax[1,0].fill_between(scat[:,3], q_obs[:,0]-np.std(q_obs[:,0]), q_obs[:,0]+np.std(q_obs[:,0]),color="blue",alpha=0.2)
    # #ax[1,0].fill_between(scat[3:5,3], q_obs[3:5,0]-np.std(q_obs[:,0]), q_obs[3:5,0]+np.std(q_obs[:,0]),facecolor="blue",alpha=0.2)

    # ax[1,1].plot(scat[:,5],q_obs[:,1],linestyle='dashed',color="red",label="660nm")
    # ax[1,1].fill_between(scat[:,5], q_obs[:,1]-0.005, q_obs[:,1]+0.005,color="red",alpha=0.2)
    # #ax[1,1].fill_between(scat[3:5,3], q_obs[3:5,1]-np.std(q_obs[:,1]), q_obs[3:5,1]+np.std(q_obs[:,1]),facecolor="red",alpha=0.2)

    # ax[1,2].plot(scat[:,6],q_obs[:,2],linestyle='dashed',color="orange",label="865nm")
    # ax[1,2].fill_between(scat[:,6], q_obs[:,2]-np.std(q_obs[:,2]), q_obs[:,2]+np.std(q_obs[:,2]),color="orange",alpha=0.2)
    
    # ax[1,0].plot(scat[:,3],q_mod[:,0],color="blue")
    # ax[1,1].plot(scat[:,5],q_mod[:,1],color="red")
    # ax[1,2].plot(scat[:,6],q_mod[:,2],color="orange")
    # ax[2,1].set_xlabel(u"\u03A9 [\u00b0]",fontsize='xx-large')   
            



