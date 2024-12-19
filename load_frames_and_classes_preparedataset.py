# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:02:44 2023

@author: LFC_01
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch
import torchvision

# path_file= r'C:\Users\LFC_01\Desktop\Szilard\Data\2023_09_28\SLM_Interferometer_alignment\M00001\input_mask_blank_screen\files\measured_phase.npy'
# measured_phase=np.load (path_file)

#%% FUNCTIONS
def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

#%%
# General data path
data_path = r'C:\Users\LFC_01\Desktop\Szilard\Data'

#Date of measurements
date='2023_10_12'
    
# Define the type of measure, so we can make a folder
data_type_measure = "SLM_Interferometer_alignment"

#Number of RUN
nRUN=1

#Classes
classes=["input_mask_blank_screen", 'input_mask_vertical_division', 'input_mask_horizontal_division', 'input_mask_checkboard_1', 'input_mask_checkboard_2', 'input_mask_checkboard_3', 'input_mask_checkboard_4']
classes_short=np.arange(np.size(classes))

#Number of frames
Nframes=nRUN*np.size(classes)

#Data to load
data_name=os.path.join('files','measured_phase.npy')
test_dir_path=os.path.join(data_path, date,data_type_measure, 'M00001', classes[0], data_name)
testmatrix=np.load(test_dir_path)
H, W=testmatrix.shape


#Inizialization of the frame matrix
measured_phase=np.zeros((H,W,Nframes))
reference_class_short=np.zeros((Nframes,1), dtype=int)
#%% DATA LOAD and plot
j=0
for i in range(nRUN):
    RUN='M0000'+ str(i+1)
    k=0
    for class_name in classes :
        dir_path=os.path.join(data_path, date,data_type_measure, RUN, class_name, data_name)
        measured_phase[:,:,j]=np.load(dir_path)
        reference_class_short[j]=k
        # #figure
        # img=plt.imshow(measured_phase[:,:,j])#cmap='Blues'
        # plt.axis('off')
        # plt.colorbar(img)
        # plt.show()
        # plt.pause(0.1)
        # print(dir_path)
        # print(j)
        #
        k+=1
        j+=1
        

#%%DATA AUGUMENTATION THROUGH CROPPING AND BINNING

# I define crop size as integer multiple Nc of the size of CIFAR-10 images, then downsize the image resolution by a factor Nc by pixel binning of. This way it is possible to feed the data to the conv net developed for classifying CIFAR-10
 
C10size=32
Nc=10

#define crop size
wc=hc=C10size*Nc #(px)

#available corner points intervals
xc_0, yc_0=[0,0]
xc_end, yc_end=[W-wc, H-hc]

#number of randomly selected cornerpoints
Nc=10

#corner points
xc=random.choices(range(xc_0, xc_end), k=Nc)
yc=random.choices(range(yc_0, yc_end), k=Nc)

#intialize crop matrix
Nframescrop=Nframes*Nc

#crop shape after binning
# newshape=np.array([np.round(wc/10), np.round(hc/10)])
# newshape=newshape.astype(int)
newshape=np.array([C10size, C10size])

measured_phase_crop=np.zeros((Nframescrop,newshape[0],newshape[1]))
reference_class_short_crop=np.zeros((Nframescrop,1), dtype=int)

#create 3D matrix of cropped frames
k=0
for i in range (Nframes):
    #figure
    img=plt.imshow(measured_phase[:,:,i])#cmap='Blues'
    plt.axis('off')
    plt.colorbar(img)
    plt.show()
    plt.pause(0.1)
    for j in range(Nc):
        crop=measured_phase[yc[j]:yc[j]+hc, xc[j]:xc[j]+wc ,i]
        crop_binned=rebin(crop,newshape)
        measured_phase_crop[k,:,:]=crop_binned
        reference_class_short_crop[k]=reference_class_short[i]
        #figure
        img=plt.imshow(measured_phase_crop[k,:,:])#cmap='Blues'
        plt.axis('off')
        plt.colorbar(img)
        plt.show()
        plt.pause(0.1)
        print(k)
        #
        k+=1

# #%%TESTS
# phase=measured_phase_crop[:,:,69]
# img=plt.imshow(phase)#cmap='Blues'
# plt.axis('off')
# plt.colorbar(img)
# plt.show()
# newshape=np.array([np.round(wc/10), np.round(hc/10)])
# newshape=newshape.astype(int)
# phase_binned=rebin(phase,newshape) 
# img=plt.imshow(phase_binned)#cmap='Blues'
# plt.axis('off')
# plt.colorbar(img)
# plt.show()

#%% CREATE DATASET FOR PYTORCHmeasured
# Datasets initialization
phase_wrapped_ds = torch.utils.data.TensorDataset(torch.from_numpy(measured_phase_crop))
phase_wrapped_train_ds, phase_wrapped_valid_ds = torch.utils.data.random_split(phase_wrapped_ds, (60,10))

# Dataloader wrappers
phase_wrapped_train_dl, phase_wrapped_valid_dl = torch.utils.data.DataLoader(phase_wrapped_train_ds), torch.utils.data.DataLoader(phase_wrapped_valid_ds)

# get some random training images
dataiter = iter(phase_wrapped_train_dl)
sampleimage = next(dataiter)
sampleimage_tensor=sampleimage[0]
# show image
img=plt.imshow(sampleimage_tensor.numpy().transpose(1, 2, 0))#cmap='Blues'
plt.axis('off')
plt.colorbar(img)
plt.show()


#NEED TO DEFINE THE LABEL TENSOR
#NEED TO FEED THE DATASET TO THE CONV NET



