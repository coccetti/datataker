# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:08:28 2022

@author: Romolo
"""
# Standard imports
import os
import numpy as np
import matplotlib.pyplot as plt
# from datetime import datetime
# This import will work only after camera drivers are installed
from pypylon import pylon
from pypylon import genicam
#
import sys
sys.path.append(os.path.dirname(__file__))
#
from phase_in_functions_2 import blank_screen, vertical_division_A, vertical_division_B, horizontal_division_A, horizontal_division_B, \
    checkerboard_1A, checkerboard_1B, checkerboard_1R, checkerboard_2A, checkerboard_2B, checkerboard_3A, checkerboard_3B, checkerboard_4A, checkerboard_4B


from acquisition_functions import acquisition
from saving_functions import make_run_folder, save_images_and_files

# import for SLM Display SDK
# You need to copy the folder holoeye from "C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v3.0\"
# to import properly the SLM drivers. This way autocompletion works better.
import holoeye
from holoeye import slmdisplaysdk


# ################################
#%% General variables
# ################################
# Define the type of measure, so we can make a folder
data_type_measure = "SLM_Interferometer_alignment"
# data_type_measure = "SLM_whatever"

# General data path
data_path = 'C:/Users/LFC_01/Desktop/Szilard/Data'

# The data will be saved in the folder data_dir
# defined after we work out the path: YYYYMMDD/type_measure/MXXXX
data_dir = ''  # Empty string at the moment

# Define camera exposure time in microseconds
camera_exposure_time = 4000  # us

# Set the used incident laser wavelength in nanometer:
laser_wavelength_nm = 532.0

# ###########################
# End of variables definition
# ###########################


#%% CAMERA INITIALIZATION
camera = pylon.InstantCamera(
    pylon.TlFactory.GetInstance().CreateFirstDevice())

camera.Open()

# enable all chunks
camera.ChunkModeActive = True

for cf in camera.ChunkSelector.Symbolics:
    camera.ChunkSelector = cf
    camera.ChunkEnable = True

camera.ExposureTime.SetValue(camera_exposure_time)  # ms

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

# %% CAMERA ACQUISITIONS 

Nrun=1000
for run in range(Nrun):
    #Create the RUN number folder to save data
    # format is M00001, M00002, ...
    run_dir = make_run_folder(data_path, data_type_measure)
    
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
    for mask in [blank_screen, vertical_division_A, vertical_division_B, horizontal_division_A, horizontal_division_B, \
        checkerboard_1A, checkerboard_1B, checkerboard_1R, checkerboard_2A, checkerboard_2B, checkerboard_3A, checkerboard_3B, checkerboard_4A, checkerboard_4B]:
    #for mask in [checkerboard_4]:
        print("\nStarting measures for", mask)
        # 1 - Set the mask
        (mask_type, phase_in) = mask(slm_data_height, slm_data_width)
        # 2 - Take the shots
        # (Mframes,Mframes_reference) = acquisition(slm, camera, mask_type, phase_shift, Nshifts,
        #                                             slm_data_width, slm_data_height,
        #                                             phase_in_reference, phase_in, Mframes_reference, Mframes)
        Mframes = acquisition(slm, camera, mask_type, phase_shift, Nshifts,
                                                    slm_data_width, slm_data_height,
                                                    phase_in, Mframes)
        ############ 
        #NOTE: to make the function 'acquisition' work properly I had to invert the outpt sequence: 
        #from (Mframes_reference, Mframes) to          (Mframes,Mframes_reference)
        ############
        
        # # 3 - Save the measures
        # save_images_and_files(run_dir, mask_type, Mframes,
        #               phase_in, phase_shift, Nshifts)
        
        # 4- compute and save measured field and phase (4-step method)
        
        selected_frames = [1, 3, 5, 7]  # frames selected from the 9 measures taken
        phase_in_selected = phase_in[selected_frames]
        Mframes_selected = Mframes[:,:,selected_frames]
        Mframes_selected = Mframes_selected.astype(np.float32)
        measured_field = (Mframes_selected[:,:,0] - Mframes_selected [:,:,2]) + 1j * \
                         (Mframes_selected[:,:,1] - Mframes_selected [:,:,3])  # flipping 2nd and 4th interferogram
        measured_phase = np.angle(measured_field)
        
        # # show measured phase
        # fig1, ax = plt.subplots()
        # img = ax.imshow(measured_phase, 'viridis')
        # # plt.colorbar(img)
        # ax.axis('off')
        # plt.pause(0.1)
        
        # save
        # First the folder for the input_mask
        mask_dir = os.path.join(run_dir, mask_type)
        if not os.path.exists(mask_dir):
            print("Making folder for the input mask:", mask_dir)
            os.makedirs(mask_dir)
        data_dir = mask_dir
        # Set the folders where you save images and np arrays
        pfile_save_dir = os.path.join(data_dir, "files")
        if not os.path.exists(pfile_save_dir):
            print("Making folder for the input mask:", pfile_save_dir)
            os.makedirs(pfile_save_dir)
        # file_path = os.path.join(pfile_save_dir, 'measured_complex_field.npy')
        # print("Saving:", file_path)
        # np.save(file_path, measured_field)
        file_path = os.path.join(pfile_save_dir, 'measured_phase.npy')
        print("Saving:", file_path)
        np.save(file_path, measured_phase)


#%% CLOSE CAMERA AND SLM
camera.Close()
slm.close()
