# -*- coding: utf-8 -*-
"""
AirMSPI_Preprocessing.py

This is a Python 3.9.13 code to read in processed AirMSPI level 1 data 
and validate that it was readin correctly

Creation Date: 2022-08-08
Last Modified: 2022-09-12

by Michael J. Garay and Clarissa M. DeLeon
(Michael.J.Garay@jpl.nasa.gov , cdeleon@arizona.edu)
"""

# Import packages and functions
from AirMSPI_ProcessData import process
import matplotlib.pyplot as plt

def visualizeprocesseed():

   data = process()
   
   for i in range(0,len(data)-1,1):
       
       data[0][i,:]
       data[1][i,:]
       data[2][i,:]
       data[3][i,:]
       data[4][i,:]
       data[5][i,:]
       data[6][i,:]
       data[7][i,:]
       data[8][i,:]
       data[9][i,:]
      

### VISUALIZATIONS TO CHECK THE RESULTS

# Set the plot area (using the concise format)

   # fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(figsize=(9,6), 
   #     nrows=2, ncols=2, dpi=120)
    
# # FIRST PLOT: INTENSITY VS. SCATTERING ANGLE
# # Plot the data
# # NOTE: To do line plots, both the x and y data must be sorted by scattering angle
        
#     ax1.scatter(scat_median[:,0],i_median[:,0],marker='o',color="indigo",s=20,label="355nm")
#     ax1.scatter(scat_median[:,1],i_median[:,1],marker='o',color="purple",s=20,label="380nm")
#     ax1.scatter(scat_median[:,2],i_median[:,2],marker='o',color="navy",s=20,label="445nm")
#     ax1.scatter(scat_median[:,3],i_median[:,3],marker='o',color="blue",s=20,label="470nm")
#     ax1.scatter(scat_median[:,4],i_median[:,4],marker='o',color="lime",s=20,label="555nm")
#     ax1.scatter(scat_median[:,5],i_median[:,5],marker='o',color="red",s=20,label="660nm")
#     ax1.scatter(scat_median[:,6],i_median[:,6],marker='o',color="magenta",s=20,label="865nm")
                
#     ax1.set_xlim(60,180)
#     ax1.set_xticks(np.arange(60,190,30))
#     ax1.set_xlabel("Scattering Angle (Deg)",fontsize=12)
    
#     ax1.set_ylim(0.0,0.6)
#     ax1.set_yticks(np.arange(0.0,0.61,0.20))
#     ax1.set_ylabel('Equivalent Reflectance',fontsize=12)
    
#     ax1.legend(loc=1,ncol=2)  # Upper right
    
# # SECOND PLOT: Q and U VS. SCATTERING ANGLE
# # Plot the data
# # NOTE: To do line plots, both the x and y data must be sorted by scattering angle
# #       Also, the index for the scattering angle and the polarized data are different
        
#     ax2.scatter(scat_median[:,3],q_median[:,0],marker='s',color="blue",s=20,label="Q-470nm")
#     ax2.scatter(scat_median[:,5],q_median[:,1],marker='s',color="red",s=20,label="Q-660nm")
#     ax2.scatter(scat_median[:,6],q_median[:,2],marker='s',color="magenta",s=20,label="Q-865nm")
    
#     ax2.scatter(scat_median[:,3],u_median[:,0],marker='D',color="blue",s=20,label="U-470nm")
#     ax2.scatter(scat_median[:,5],u_median[:,1],marker='D',color="red",s=20,label="U-660nm")
#     ax2.scatter(scat_median[:,6],u_median[:,2],marker='D',color="magenta",s=20,label="U-865nm")
                  
#     ax2.set_xlim(60,180)
#     ax2.set_xticks(np.arange(60,190,30))
#     ax2.set_xlabel("Scattering Angle (Deg)",fontsize=12)
    
#     ax2.set_ylim(-0.1,0.1)
#     ax2.set_yticks(np.arange(-0.1,0.11,0.05))
#     ax2.set_ylabel('Polarized Reflectance',fontsize=12)

#     ax2.plot([60,180],[0.0,0.0],color="black",linewidth=1)  # Line at zero
    
#     ax2.legend(loc=1,ncol=2)  # Upper right

# # THIRD PLOT: Ipol VS. SCATTERING ANGLE
# # Plot the data
# # NOTE: To do line plots, both the x and y data must be sorted by scattering angle
        
#     ax3.scatter(scat_median[:,3],ipol_median[:,0],marker='H',color="blue",s=20,label="470nm")
#     ax3.scatter(scat_median[:,5],ipol_median[:,1],marker='H',color="red",s=20,label="660nm")
#     ax3.scatter(scat_median[:,6],ipol_median[:,2],marker='H',color="magenta",s=20,label="865nm")
    
#     ax3.set_xlim(60,180)
#     ax3.set_xticks(np.arange(60,190,30))
#     ax3.set_xlabel("Scattering Angle (Deg)",fontsize=12)
    
#     ax3.set_ylim(0.0,0.1)
#     ax3.set_yticks(np.arange(0.0,0.11,0.02))
#     ax3.set_ylabel('Polarized Equivalent Reflectance',fontsize=12)
#     ax3.legend(loc=1,ncol=2)
    
# # FOURTH PLOT: DoLP VS. SCATTERING ANGLE
# # Plot the data
# # NOTE: To do line plots, both the x and y data must be sorted by scattering angle
        
#     ax4.scatter(scat_median[:,3],ipol_median[:,0],marker='^',color="blue",s=20,label="470nm")
#     ax4.scatter(scat_median[:,5],ipol_median[:,1],marker='^',color="red",s=20,label="660nm")
#     ax4.scatter(scat_median[:,6],ipol_median[:,2],marker='^',color="magenta",s=20,label="865nm")
    
#     ax4.set_xlim(60,180)
#     ax4.set_xticks(np.arange(60,190,30))
#     ax4.set_xlabel("Scattering Angle (Deg)",fontsize=12)
    
#     ax4.set_ylim(0.0,0.1)
#     ax4.set_yticks(np.arange(0.0,0.11,0.02))
#     ax4.set_ylabel('DoLP [Decimal]',fontsize=12)
    
#     ax4.legend(loc=1,ncol=2)

# # Tight layout

#     plt.tight_layout()
    
# # Show the plot
    
#     plt.show()
    
# # Close the plot
        
#     plt.close()

###----------------------- END MAIN FUNCTION-------------------------------###