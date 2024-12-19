# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:19:52 2024

@author: LFC_01
"""

# Standard imports
import os
import numpy as np
import matplotlib.pyplot as plt
# from datetime import datetime
#
from phase_in_functions_2 import blank_screen, vertical_division_A, vertical_division_B, horizontal_division_A, horizontal_division_B, \
    checkerboard_1A, checkerboard_1B, checkerboard_1R, checkerboard_2A, checkerboard_2B, checkerboard_3A, checkerboard_3B, checkerboard_4A,         checkerboard_4B

# %% INPUT PHASE MASK
# Read from slm, we know dataHeight==1080
slm_data_height = 1080
# Read from slm, we know dataWidth==1920
slm_data_width = 1920

for mask in [blank_screen, vertical_division_A, vertical_division_B, horizontal_division_A, horizontal_division_B, \
    checkerboard_1A, checkerboard_1B, checkerboard_1R, checkerboard_2A, checkerboard_2B, checkerboard_3A, checkerboard_3B, checkerboard_4A, checkerboard_4B]:
#for mask in [checkerboard_4]:
    #print("\nStarting measures for", mask)
    # 1 - Set the mask
    (mask_type, phase_in) = mask(slm_data_height, slm_data_width)
    #figure
    img=plt.imshow(phase_in)#cmap='Blues'
    plt.axis('off')
    plt.colorbar(img)
    plt.show()
    plt.pause(0.1)