# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:15:07 2023

@author: ULTRASIP_1
"""

#Import Libraries 
import logging
from hamamatsu.dcam import dcam, Stream, copy_frame

def takeimage(nb_frames,exp):
    
    logging.basicConfig(level=logging.INFO)

    with dcam:
        camera = dcam[0]
        with camera:
            print(camera.info)
            print(camera['image_width'].value, camera['image_height'].value)

            # Simple acquisition example
            camera["exposure_time"] = exp
            with Stream(camera, nb_frames) as stream:
                    logging.info("start acquisition")
                    camera.start()
                    for i, frame_buffer in enumerate(stream):
                        frame = copy_frame(frame_buffer)
                        logging.info(f"acquired frame #%d/%d: %s", (i+1), nb_frames, frame)                
                        logging.info("finished acquisition")
                        camera.stop()
                    return frame
