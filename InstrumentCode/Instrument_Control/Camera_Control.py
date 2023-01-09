# -*- coding: utf-8 -*-
"""
@author: C.M.DeLeon
This script is to control the ULTRASIP Hamamatsu camera
from example: https://pypi.org/project/hamamatsu/ 

NOTE: Must download API specific to your frame grabber 
from: https://dcam-api.com/downloads/
"""
#Last edit: 06.28.2022

#Import Libraries 
import logging
from hamamatsu.dcam import dcam, Stream, copy_frame
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)

with dcam:
    camera = dcam[0]
    with camera:
        print(camera.info)
        print(camera['image_width'].value, camera['image_height'].value)

        # Simple acquisition example
        nb_frames = 5
        camera["exposure_time"] = 1e-6
        with Stream(camera, nb_frames) as stream:
                logging.info("start acquisition")
                camera.start()
                for i, frame_buffer in enumerate(stream):
                    frame = copy_frame(frame_buffer)
                    logging.info(f"acquired frame #%d/%d: %s", i+1, nb_frames, frame)
                logging.info("finished acquisition")
                
plt.imshow(frame)
plt.colorbar()