import numpy as np


def a_incenter_b(a, b):
    # placing matrix a in the center of matrix b
    db = b.shape
    da = a.shape
    lower_y = (db[0] // 2) - (da[0] // 2)
    upper_y = (db[0] // 2) + (da[0] // 2)
    lower_x = (db[1] // 2) - (da[1] // 2)
    upper_x = (db[1] // 2) + (da[1] // 2)
    b[lower_y:upper_y, lower_x:upper_x] = a
    return b


def blank_screen(slm_data_height, slm_data_width):
    # Full blank screen
    mask_type = "input_mask_blank_screen"
    phaseIn = np.zeros((slm_data_height, slm_data_width))
    # plt.imshow(phaseIn)
    return mask_type, phaseIn


def vertical_division(slm_data_height, slm_data_width):
    # divide screen vertical
    mask_type = "input_mask_vertical_division"
    phaseA = 0
    phaseB = np.pi
    screenDivider = 0.5
    phaseIn = np.zeros((slm_data_height, slm_data_width))
    screenElement = np.int32(np.floor(slm_data_width * screenDivider))
    phaseIn[:, 0:screenElement] = phaseA
    phaseIn[:, screenElement + 1:slm_data_width] = phaseB
    # plt.imshow(phaseIn)
    return mask_type, phaseIn


def horizontal_division(slm_data_height, slm_data_width):
    # divide screen horizontal
    mask_type = "input_mask_horizontal_division"
    phaseA = 0
    phaseB = np.pi
    screenDivider = 0.5
    phaseIn = np.zeros((slm_data_height, slm_data_width))
    screenElement = np.int32(np.floor(slm_data_height * screenDivider))
    phaseIn[0:screenElement, :] = phaseA
    phaseIn[screenElement + 1:slm_data_height, :] = phaseB
    # plt.imshow(phaseIn)
    return mask_type, phaseIn


def checkerboard_1(slm_data_height, slm_data_width):
    # checkerboard (define square single field size (px) of checkerboard with respect to slm width, checkerboard centered in the slm screen )
    mask_type = "input_mask_checkboard_1"
    nsingleV = 1  ##number of fields single type, vertical direction
    nsingleH = 2  ##number of fields single type, horizontal direction
    # n=nsingleV*2 #number of fields both type
    # npixelsingle=np.int16(np.floor(dataHeight/n))
    n = nsingleH * 2  # number of fields both type
    npixelsingle = np.int16(np.floor(slm_data_width / n))
    Mceckerboard = np.kron([[1, 0] * nsingleH, [0, 1] * nsingleH] * nsingleV, np.ones((npixelsingle, npixelsingle)))
    phaseA = 0
    phaseB = np.pi
    phaseIn_0 = np.zeros((slm_data_height, slm_data_width))
    phaseIn = a_incenter_b(phaseB * Mceckerboard, phaseIn_0)
    # plt.imshow(phaseIn)
    return mask_type, phaseIn


def checkerboard_2(slm_data_height, slm_data_width):
    # checkerboard (define square single field size (px) of checkerboard with respect to slm width, checkerboard centered in the slm screen )
    mask_type = "input_mask_checkboard_2"
    nsingleV = 2  ##number of fields single type, vertical direction
    nsingleH = 4  ##number of fields single type, horizontal direction
    # n=nsingleV*2 #number of fields both type
    # npixelsingle=np.int16(np.floor(dataHeight/n))
    n = nsingleH * 2  # number of fields both type
    npixelsingle = np.int16(np.floor(slm_data_width / n))
    Mceckerboard = np.kron([[1, 0] * nsingleH, [0, 1] * nsingleH] * nsingleV, np.ones((npixelsingle, npixelsingle)))
    phaseA = 0
    phaseB = np.pi
    phaseIn_0 = np.zeros((slm_data_height, slm_data_width))
    phaseIn = a_incenter_b(phaseB * Mceckerboard, phaseIn_0)
    # plt.imshow(phaseIn)
    return mask_type, phaseIn


def checkerboard_3(slm_data_height, slm_data_width):
    # checkerboard (define square single field size (px) of checkerboard with respect to slm width, checkerboard centered in the slm screen )
    mask_type = "input_mask_checkboard_3"
    nsingleV = 3  ##number of fields single type, vertical direction
    nsingleH = 6  ##number of fields single type, horizontal direction
    # n=nsingleV*2 #number of fields both type
    # npixelsingle=np.int16(np.floor(dataHeight/n))
    n = nsingleH * 2  # number of fields both type
    npixelsingle = np.int16(np.floor(slm_data_width / n))
    Mceckerboard = np.kron([[1, 0] * nsingleH, [0, 1] * nsingleH] * nsingleV, np.ones((npixelsingle, npixelsingle)))
    phaseA = 0
    phaseB = np.pi
    phaseIn_0 = np.zeros((slm_data_height, slm_data_width))
    phaseIn = a_incenter_b(phaseB * Mceckerboard, phaseIn_0)
    # plt.imshow(phaseIn)
    return mask_type, phaseIn


def checkerboard_4(slm_data_height, slm_data_width):
    # checkerboard (define square single field size (px) of checkerboard with respect to slm width, checkerboard centered in the slm screen )
    mask_type = "input_mask_checkboard_4"
    nsingleV = 4  ##number of fields single type, vertical direction
    nsingleH = 8  ##number of fields single type, horizontal direction
    # n=nsingleV*2 #number of fields both type
    # npixelsingle=np.int16(np.floor(dataHeight/n))
    n = nsingleH * 2  # number of fields both type
    npixelsingle = np.int16(np.floor(slm_data_width / n))
    Mceckerboard = np.kron([[1, 0] * nsingleH, [0, 1] * nsingleH] * nsingleV, np.ones((npixelsingle, npixelsingle)))
    phaseA = 0
    phaseB = np.pi
    phaseIn_0 = np.zeros((slm_data_height, slm_data_width))
    phaseIn = a_incenter_b(phaseB * Mceckerboard, phaseIn_0)
    # plt.imshow(phaseIn)
    return mask_type, phaseIn