# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:08:28 2022

@author: Romolo
"""
# Standard imports
# import os
import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
from phase_in_functions import blank_screen, vertical_division, horizontal_division
from camera_functions import camera_shots
from saving_functions import make_run_folder, save_measures

# This import will work only after camera drivers are installed
# from pypylon import pylon
# from pypylon import genicam

# import for SLM Display SDK
# You need to copy the folder holoeye from "C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v3.0\"
# to import properly the SLM drivers. This way autocompletion works better.
import holoeye
from holoeye import slmdisplaysdk

# ################################
# General variables
# ################################
# Define the type of measure, so we can make a folder
data_type_measure = "SLM_Interferometer_alignment"
# data_type_measure = "SLM_whatever"

# General data path
data_path = 'C:/Users/LFC_01/Desktop/Szilard/Data'

# The data will be saved in the folder data_dir
# defined after we work out the path: YYYYMMDD/type_measure/MXXXX
data_dir = ''  # Empty string at the moment

# Define camera exposure time in milliseconds
camera_exposure_time = 50  # ms

# Set the used incident laser wavelength in nanometer:
laser_wavelength_nm = 532.0

# ###########################
# End of variables definition
# ###########################

# Create the RUN number folder to save data
# format is M00001, M00002, ...
run_dir = make_run_folder(data_path, data_type_measure)


# # %% CAMERA INITIALIZATION
# camera = pylon.InstantCamera(
#     pylon.TlFactory.GetInstance().CreateFirstDevice())
#
# camera.Open()
#
# # enable all chunks
# camera.ChunkModeActive = True
#
# for cf in camera.ChunkSelector.Symbolics:
#     camera.ChunkSelector = cf
#     camera.ChunkEnable = True
#
# camera.ExposureTime.SetValue(camera_exposure_time)  # ms

# %% SLM INITIALIZATION
# Initializes the SLM library
slm = slmdisplaysdk.SLMInstance()

# Check if the library implements the required version
if not slm.requiresVersion(3):
    print("SLM library has NOT the required version!! Exiting...")
    exit(1)

# Detect SLMs and open a window on the selected SLM
error = slm.open()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Open the SLM preview window in non-scaled mode:
# Please adapt the file showSLMPreview.py if preview window
# is not at the right position or even not visible.

# from showSLMPreview import showSLMPreview
# showSLMPreview(slm, scale=1.0)

# %% LOAD SLM PHASE COMPENSATION FILE
# Load and use the Correction Function file for the SLM
wavefrontfile = (
    r'C:\Users\LFC_01\Documents\SLM_PLUTO_MATERIAL\Wavefront_Correction_Function\U.14-2040-182427-2X-00-05_7020-1 6010-1086.h5')
error = slm.wavefrontcompensationLoad(wavefrontfile, laser_wavelength_nm,
                                      slmdisplaysdk.WavefrontcompensationFlags.NoFlag, 0, 0)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# %% INPUT PHASE MASK
# Read from slm, we know dataHeight==1080
slm_data_height = slm.height_px
# Read from slm, we know dataWidth==1920
slm_data_width = slm.width_px

# Create phase_in for the Reference blank screen
phase_in_reference = np.zeros((slm_data_height, slm_data_width))

# %% CAMERA ACQUISITIONS OVER A SERIES OF UNIFORM PHASE SHIFTS
# ######################
# phase shifts
Nshifts = 9
phase_shift = np.linspace(0, 2 * np.pi, num=Nshifts)

# Camera data
camera_frame_width = 1920
camera_frame_height = 1200
Mframes = np.zeros((camera_frame_height, camera_frame_width, Nshifts), dtype=np.uint8)
Mframes_reference = np.zeros((camera_frame_height, camera_frame_width, Nshifts), dtype=np.uint8)

# Call the functions to compute phase_in
# Main loop for the mask names
for mask in [blank_screen, vertical_division, horizontal_division]:
    print("\nStarting measures for", mask)
    # 1 - Set the mask
    (mask_type, phase_in) = mask(slm_data_height, slm_data_width)
    # 2 - Take the shots
    (Mframes_reference, Mframes) = camera_shots(mask_type, phase_shift, Nshifts,
                                                slm_data_width, slm_data_height,
                                                phase_in_reference, phase_in, Mframes_reference, Mframes)
    # 3 - Save the measures
    save_measures(run_dir, mask_type, Mframes_reference, Mframes,
                  phase_in_reference, phase_in, phase_shift, Nshifts)


# # %% CLOSE CAMERA AND SLM
# camera.Close()
# slm.close()
