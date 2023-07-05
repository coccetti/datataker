# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:08:28 2022

@author: Romolo
"""
# Standard imports
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# This import will work only after camera drivers are installed
from pypylon import pylon
from pypylon import genicam

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

# create folders to save data
# should be here
# but I will move it later
# at the moment, it's at the bottom of the file


# %% FUNCTION
# placing matrix a in the center of matrix b
def a_incenter_b(a, b):
    db = b.shape
    da = a.shape
    lower_y = (db[0] // 2) - (da[0] // 2)
    upper_y = (db[0] // 2) + (da[0] // 2)
    lower_x = (db[1] // 2) - (da[1] // 2)
    upper_x = (db[1] // 2) + (da[1] // 2)
    b[lower_y:upper_y, lower_x:upper_x] = a
    return b


# %% CAMERA INITIALIZATION
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
wavefrontfile = (
    r'C:\Users\LFC_01\Documents\SLM_PLUTO_MATERIAL\Wavefront_Correction_Function\U.14-2040-182427-2X-00-05_7020-1 6010-1086.h5')
error = slm.wavefrontcompensationLoad(wavefrontfile, laser_wavelength_nm,
                                      slmdisplaysdk.WavefrontcompensationFlags.NoFlag, 0, 0)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# %% INPUT PHASE MASK
dataWidth = slm.width_px
dataHeight = slm.height_px
# dataWidth = 100
# dataHeight = 60

# Reference blank screen
phaseIn_reference = np.zeros((dataHeight, dataWidth))

# blank screen (phase=0)
# mask_type = "input_mask_blank_screen"
# phaseIn = np.zeros((dataHeight, dataWidth))
# plt.imshow(phaseIn)

# # divide screen vertical
# mask_type = "input_mask_vertical_division"
# phaseA=0
# phaseB=np.pi
# screenDivider = 0.5
# phaseIn=np.zeros((dataHeight,dataWidth))
# screenElement=np.int32(np.floor(dataWidth*screenDivider))
# phaseIn[:,0:screenElement]=phaseA
# phaseIn[:,screenElement+1:dataWidth]=phaseB
# plt.imshow(phaseIn)


# divide screen horizontal
mask_type = "input_mask_horizontal_division"
phaseA=0
phaseB=np.pi
screenDivider = 0.5
phaseIn=np.zeros((dataHeight,dataWidth))
screenElement=np.int32(np.floor(dataHeight*screenDivider))
phaseIn[0:screenElement, :]=phaseA
phaseIn[screenElement+1:dataHeight,:]=phaseB
plt.imshow(phaseIn)

# #checkerboard (define square single field size (px) of checkerboard with respect to slm width, checkerboard centered in the slm screen )
# mask_type = "input_mask_checkboard_1"
# nsingleV=1 ##number of fields single type, vertical direction
# nsingleH=2 ##number of fields single type, horizontal direction
# # n=nsingleV*2 #number of fields both type
# # npixelsingle=np.int16(np.floor(dataHeight/n))
# n=nsingleH*2 #number of fields both type
# npixelsingle=np.int16(np.floor(dataWidth/n))
# Mceckerboard=np.kron([[1, 0] * nsingleH, [0, 1] * nsingleH] * nsingleV, np.ones((npixelsingle, npixelsingle)))
# phaseA=0
# phaseB=np.pi
# phaseIn_0=np.zeros((dataHeight,dataWidth))
# phaseIn= a_incenter_b(phaseB*Mceckerboard,phaseIn_0)
# plt.imshow(phaseIn)

# ######################
# %% CAMERA ACQUISITIONS OVER A SERIES OF UNIFORM PHASE SHIFTS
# ######################
# phase shifts
Nshifts = 9
phaseshift = np.linspace(0, 2 * np.pi, num=Nshifts)

# Camera data
frameWidth = 1920
frameHeight = 1200
Mframes = np.zeros((frameHeight, frameWidth, Nshifts), dtype=np.uint8)
Mframes_reference = np.zeros((frameHeight, frameWidth, Nshifts), dtype=np.uint8)

for i in range(Nshifts):
    print("Phase shift:", f"{phaseshift[i]:.4f}")

    # Code for the reference
    print("  Taking shot for reference")
    phaseData = slmdisplaysdk.createFieldSingle(dataWidth, dataHeight) + phaseIn_reference + phaseshift[i]
    # error = slm.wavefrontcompensationLoad(phaseData, laser_wavelength_nm, slmdisplaysdk.WavefrontcompensationFlags.NoFlag, 0, 0)
    # assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
    error = slm.showPhasevalues(phaseData)  # display phase values on the SLM
    assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
    result = camera.GrabOne(100)  # grab frame file on the camera
    Mframes_reference[:, :, i] = result.Array  # extract numerical matrix and build 3D frame matrix

    # Code for the phaseIn selected
    print("  Taking shot for", mask_type)
    phaseData = slmdisplaysdk.createFieldSingle(dataWidth, dataHeight) + phaseIn + phaseshift[i]
    # error = slm.wavefrontcompensationLoad(phaseData, laser_wavelength_nm, slmdisplaysdk.WavefrontcompensationFlags.NoFlag, 0, 0)
    # assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
    error = slm.showPhasevalues(phaseData)  # display phase values on the SLM
    assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
    result = camera.GrabOne(100)  # grab frame file on the camera
    Mframes[:, :, i] = result.Array  # extract numerical matrix and build 3D frame matrix


# %% CLOSE CAMERA AND SLM
camera.Close()
slm.close()


# %% SHOWING AND SAVING IMAGES
# Before saving, let's make the necessary folders
# in order to remember the input mask this lines are at the bottom of the file
today = datetime.now()
data_day_dir = os.path.join(data_path, today.strftime('%Y_%m_%d'))
data_measure_dir = os.path.join(data_day_dir, data_type_measure)
mask_dir = os.path.join(data_measure_dir, mask_type)
# Make folder for the day
if not os.path.exists(data_day_dir):
    print("Making folder for the day:", data_day_dir)
    os.makedirs(data_day_dir)
# Make folder for the type of measure
if not os.path.exists(data_measure_dir):
    print("Making folder for the type of measure:", data_measure_dir)
    os.makedirs(data_measure_dir)
# Make another for the input_mask
if not os.path.exists(mask_dir):
    print("Making folder for the input mask:", mask_dir)
    os.makedirs(mask_dir)
# Make folder for the run
for ii in range(1, 9999):
    data_dir = os.path.join(mask_dir + f"/M{ii:05d}")
    if not os.path.exists(data_dir):
        print("Making folder for the current run:", data_dir)
        os.mkdir(data_dir)
        break
# Set the folders where you save images and np arrays
image_save_dir = os.path.join(data_dir, "images")
pfile_save_dir = os.path.join(data_dir, "files")
# Make folders
os.mkdir(image_save_dir)
os.mkdir(pfile_save_dir)


for i in range(Nshifts):
    fig1, ax = plt.subplots()
    img = ax.imshow(Mframes[:, :, i], 'viridis')
    # plt.colorbar(img)
    ax.axis('off')
    #
    imagename = 'frame' + np.str(i) + '.png'
    file_path = os.path.join(image_save_dir, imagename)
    print("Saving:", file_path)
    plt.savefig(file_path)

for i in range(Nshifts):
    fig1, ax = plt.subplots()
    img = ax.imshow(Mframes_reference[:, :, i], 'viridis')
    # plt.colorbar(img)
    ax.axis('off')
    #
    imagename = 'frame' + np.str(i) + '_reference.png'
    file_path = os.path.join(image_save_dir, imagename)
    print("Saving:", file_path)
    plt.savefig(file_path)

# %% SAVIG FILES
# save frames matrix
file_path = os.path.join(pfile_save_dir, 'frames.npy')
print("Saving:", file_path)
np.save(file_path, Mframes)
file_path = os.path.join(pfile_save_dir, 'frames_reference.npy')
print("Saving:", file_path)
np.save(file_path, Mframes_reference)

# save input phase mask
file_path = os.path.join(pfile_save_dir, 'phasein.npy')
print("Saving:", file_path)
np.save(file_path, phaseIn)
file_path = os.path.join(pfile_save_dir, 'phasein_reference.npy')
print("Saving:", file_path)
np.save(file_path, phaseIn_reference)

# save input phase shift
file_path = os.path.join(pfile_save_dir, 'phaseshifts.npy')
print("Saving:", file_path)
np.save(file_path, phaseshift)
