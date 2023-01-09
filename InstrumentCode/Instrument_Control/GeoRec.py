# -*- coding: utf-8 -*-
"""
Geometry Reconciliation 
"""

import numpy as np
import math

def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))

north = np.array([-0.5, 0.5, 0])
view = np.array([-0.5 ,-0.25, 0.25]) 
zenith = np.array([0.5, 0,-0.5])

normal_grasp =  np.cross(zenith,north)/magnitude(np.cross(zenith,north))
#print(normal_grasp)

vertical_grasp = np.cross(normal_grasp, view)/magnitude(np.cross(normal_grasp, view))
#print(vertical_grasp)

horizontal_grasp = np.cross(vertical_grasp, view)/magnitude(np.cross(vertical_grasp, view))
#print(horizontal_grasp)


normal_air =  np.cross(zenith,view)/magnitude(np.cross(zenith,view))
#print(normal_air)

vertical_air = np.cross(normal_air, view)/magnitude(np.cross(normal_air, view))
#print(vertical_air)

horizontal_air = np.cross(vertical_air, view)/magnitude(np.cross(vertical_air, view))
#print(horizontal_air)

#print(math.acos(np.dot(vertical_grasp,vertical_air)))
#print(math.acos(np.dot(horizontal_grasp,horizontal_air)))

angdifv = math.acos(np.dot(vertical_grasp,vertical_air))
angdifh = math.acos(np.dot(horizontal_grasp,horizontal_air))

signv = np.sign(np.dot(view,np.cross(vertical_grasp,vertical_air)))
signh = np.sign(np.dot(view,np.cross(horizontal_grasp,horizontal_air)))

#print(signv)
#print(signh)

rotangv = angdifv*signv 
rotangh = angdifh*signh

print(rotangv,rotangh)
