# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:02:44 2023

@author: LFC_01
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# path_file= r'C:\Users\LFC_01\Desktop\Szilard\Data\2023_09_28\SLM_Interferometer_alignment\M00001\input_mask_blank_screen\files\measured_phase.npy'
# measured_phase=np.load (path_file)



# General data path
data_path = r'C:\Users\LFC_01\Desktop\Szilard\Data'

#Date of measurements
date='2023_10_12'
    
# Define the type of measure, so we can make a folder
data_type_measure = "SLM_Interferometer_alignment"

#Number of RUN
nRUN=5

#Classes
classes=["input_mask_blank_screen", 'input_mask_vertical_division', 'input_mask_horizontal_division', 'input_mask_checkboard_1', 'input_mask_checkboard_2', 'input_mask_checkboard_3', 'input_mask_checkboard_4']
classes_short=np.arange(np.size(classes))

#Data to load
data_name=os.path.join('files','measured_phase.npy')
test_dir_path=os.path.join(data_path, date,data_type_measure, 'M00001', classes[0], data_name)
testmatrix=np.load(test_dir_path)

#Inizialization of the frame matrix
measured_phase=np.zeros((np.shape(testmatrix)[0],np.shape(testmatrix)[1],nRUN*np.size(classes)))
reference_class_short=np.zeros((nRUN*np.size(classes),1), dtype=int)
#%% DATA LOAD and plot
j=0
for i in range(nRUN):
    RUN='M0000'+ str(i+1)
    k=0
    for class_name in classes :
        dir_path=os.path.join(data_path, date,data_type_measure, RUN, class_name, data_name)
        measured_phase[:,:,j]=np.load(dir_path)
        reference_class_short[j]=k
        #figure
        img=plt.imshow(measured_phase[:,:,j])#cmap='Blues'
        plt.axis('off')
        plt.colorbar(img)
        plt.show()
        plt.pause(0.1)
        print(dir_path)
        print(j)
        k+=1
        j+=1
        
