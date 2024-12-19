# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:02:44 2023

@author: LFC_01
"""

import numpy as np
import matplotlib.pyplot as plt
import os
# import random
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

#%% FUNCTIONS

def resizekeepAR(frame, newheight): 
#resize the input image  array for the new height by keeping the aspect ratio
    hpercent = (newheight/float(frame.size[1]))
    newwidth = int((float(frame.size[0])*float(hpercent)))
    frame = frame.resize((newwidth, newheight))
    return frame
    
#%% CLASSES

#%%
# General data path
data_path = r'C:\Users\LFC_01\Desktop\Szilard\Data'

#Date of measurements
date='2023_11_08'
    
# Define the type of measure, so we can make a folder
data_type_measure = "SLM_Interferometer_alignment"

#Number of RUN
nRUN=1

#Classes
classes=["input_mask_blank_screen", 'input_mask_vertical_division', 'input_mask_horizontal_division', 'input_mask_checkboard_1', 'input_mask_checkboard_2', 'input_mask_checkboard_3', 'input_mask_checkboard_4']
#classes_short=np.arange(np.size(classes))

#Number of frames
Nframes=nRUN*np.size(classes)

#Data to load
data_name_wrappedphase=os.path.join('files','measured_phase.npy')
dir_path_test=os.path.join(data_path, date,data_type_measure, 'M00001', classes[0], data_name_wrappedphase)
wrappedphase_test=np.load(dir_path_test)
H_pm, W_pm=wrappedphase_test.shape #size of phase matrix
#
data_name_phaseslm=os.path.join('files','phasein.npy')
dir_path_test=os.path.join(data_path, date,data_type_measure, 'M00001', classes[0], data_name_phaseslm)
phaseslm_test=np.load(dir_path_test)
H_pslm, W_pslm=phaseslm_test.shape #size of ground truth matrix

#Inizialization of the entire matrix of wrapped phases
wrappedphase=np.zeros((H_pm, W_pm, Nframes))
#Inizialization of the entire matrix of groud truth phases
phaseslm=np.zeros((H_pslm,W_pslm, Nframes))

#reference_class_short=np.zeros((Nframes,1), dtype=int)
#%% DATA LOAD and plot
j=0
for i in range(nRUN):
    RUN='M0000'+ str(i+1)
    if i+1>9:
        RUN='M000'+ str(i+1)
        if i+1>99:
            RUN='M00'+ str(i+1)
    k=0
    for class_name in classes :
        dir_path=os.path.join(data_path, date,data_type_measure, RUN, class_name, data_name_wrappedphase)
        wrappedphase[:,:,j]=np.load(dir_path)
        #
        dir_path=os.path.join(data_path, date,data_type_measure, RUN, class_name, data_name_phaseslm)
        phaseslm[:,:,j]=np.load(dir_path)
        #reference_class_short[j]=k
        #
        #figure
        img=plt.imshow(wrappedphase[:,:,j])#cmap='Blues'
        plt.axis('off')
        plt.colorbar(img)
        plt.show()
        plt.pause(0.1)
        #figure
        img=plt.imshow(phaseslm[:,:,j])#cmap='Blues'
        plt.axis('off')
        plt.colorbar(img)
        plt.show()
        plt.pause(0.1)
        # print(dir_path)
        # print(j)
        #
        k+=1
        j+=1
        
#%%TEST
i=3
A=wrappedphase[:,:,i]
img=plt.imshow(A)
B=phaseslm[:,:,i]
img=plt.imshow(B)
#BB=resizekeepAR(B,H_pm)

# import PIL
# from PIL import Image

frame=B
newheight=H_pm
hpercent = (newheight/float(frame.shape[0]))
newwidth = int((float(frame.shape[1])*float(hpercent)))
frameframe = frame.resize((newwidth,newheight))