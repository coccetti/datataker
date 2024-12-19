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




def vertical_division_A(slm_data_height, slm_data_width):
    # divide screen vertical
    mask_type = "input_mask_vertical_division_A"
    phaseA = 0
    phaseB = np.pi
    screenDivider = 0.5
    phaseIn = np.zeros((slm_data_height, slm_data_width))
    screenElement = np.int32(np.floor(slm_data_width * screenDivider))
    phaseIn[:, 0:screenElement] = phaseA
    phaseIn[:, screenElement + 1:slm_data_width] = phaseB
    # plt.imshow(phaseIn)
    return mask_type, phaseIn

def vertical_division_B(slm_data_height, slm_data_width):
    # divide screen vertical
    mask_type = "input_mask_vertical_division_B"
    phaseA = 0
    phaseB = np.pi
    screenDivider = 0.5
    phaseIn = np.zeros((slm_data_height, slm_data_width))
    screenElement = np.int32(np.floor(slm_data_width * screenDivider))
    phaseIn[:, 0:screenElement] = phaseB
    phaseIn[:, screenElement + 1:slm_data_width] = phaseA
    # plt.imshow(phaseIn)
    return mask_type, phaseIn





def horizontal_division_A(slm_data_height, slm_data_width):
    # divide screen horizontal
    mask_type = "input_mask_horizontal_division_A"
    phaseA = 0
    phaseB = np.pi
    screenDivider = 0.5
    phaseIn = np.zeros((slm_data_height, slm_data_width))
    screenElement = np.int32(np.floor(slm_data_height * screenDivider))
    phaseIn[0:screenElement, :] = phaseA
    phaseIn[screenElement + 1:slm_data_height, :] = phaseB
    # plt.imshow(phaseIn)
    return mask_type, phaseIn

def horizontal_division_B(slm_data_height, slm_data_width):
    # divide screen horizontal
    mask_type = "input_mask_horizontal_division_B"
    phaseA = 0
    phaseB = np.pi
    screenDivider = 0.5
    phaseIn = np.zeros((slm_data_height, slm_data_width))
    screenElement = np.int32(np.floor(slm_data_height * screenDivider))
    phaseIn[0:screenElement, :] = phaseB
    phaseIn[screenElement + 1:slm_data_height, :] = phaseA
    # plt.imshow(phaseIn)
    return mask_type, phaseIn





def checkerboard_1A(slm_data_height, slm_data_width):
    # checkerboard (define square single field size (px) of checkerboard with respect to slm width, checkerboard centered in the slm screen )
    mask_type = "input_mask_checkboard_1A"
    nsingleV = 1  ##number of fields single type, vertical direction
    nsingleH = 2  ##number of fields single type, horizontal direction
    # n=nsingleV*2 #number of fields both type
    # npixelsingle=np.int16(np.floor(dataHeight/n))
    n = nsingleH * 2  # number of fields both type
    npixelsingle = np.int16(np.floor(slm_data_width / n))
    Mceckerboard = np.kron([[1, 0] * nsingleH, [0, 1] * nsingleH] * nsingleV, np.ones((npixelsingle, npixelsingle)))
    phaseA = 0
    phaseB = np.pi
    phaseIn_0 = phaseA + np.zeros((slm_data_height, slm_data_width))
    phaseIn = a_incenter_b(phaseB * Mceckerboard, phaseIn_0)
    # plt.imshow(phaseIn)
    return mask_type, phaseIn

def checkerboard_1B(slm_data_height, slm_data_width):
    # checkerboard (define square single field size (px) of checkerboard with respect to slm width, checkerboard centered in the slm screen )
    mask_type = "input_mask_checkboard_1B"
    nsingleV = 1  ##number of fields single type, vertical direction
    nsingleH = 2  ##number of fields single type, horizontal direction
    # n=nsingleV*2 #number of fields both type
    # npixelsingle=np.int16(np.floor(dataHeight/n))
    n = nsingleH * 2  # number of fields both type
    npixelsingle = np.int16(np.floor(slm_data_width / n))
    Mceckerboard = np.kron([[1, 0] * nsingleH, [0, 1] * nsingleH] * nsingleV, np.ones((npixelsingle, npixelsingle)))
    phaseA = 0
    phaseB = np.pi
    phaseIn_0 = phaseB + np.zeros((slm_data_height, slm_data_width))
    phaseIn = a_incenter_b(phaseB *(-1)*(Mceckerboard-1), phaseIn_0)  
    # img=plt.imshow((-1)*(Mceckerboard-1))#cmap='Blues'
    # plt.axis('off')
    # plt.colorbar(img)
    # plt.show() 
    return mask_type, phaseIn

def checkerboard_1R(slm_data_height, slm_data_width):
    # checkerboard (define square single field size (px) of checkerboard with respect to slm width, checkerboard centered in the slm screen )
    mask_type = "input_mask_checkboard_1R"
    nsingleV = 1  ##number of fields single type, vertical direction
    nsingleH = 2  ##number of fields single type, horizontal direction
    # n=nsingleV*2 #number of fields both type
    # npixelsingle=np.int16(np.floor(dataHeight/n))
    n = nsingleH * 2  # number of fields both type, horizontal direction
    npixelsingle = np.int16(np.floor(slm_data_width / n))
    Mceckerboard = np.kron([[1, 0] * nsingleH, [0, 1] * nsingleH] * nsingleV, np.ones((npixelsingle, npixelsingle)))
    
    vertex_y=(np.arange(1, nsingleV*2+1)*npixelsingle)-1 # vertical index of vertex points between checkerboard blocks
    vertex_x=(np.arange(1, nsingleH*2+1)*npixelsingle)-1 # horizontal index of vertex points between checkerboard blocks
    k=1
    h=0
    for j in range(vertex_x[k]+1, vertex_x[k+1]+1):  #block indicated by the vertwx indexed (k,h) is set to value 1 (instead of 0)
        for i in range(vertex_y[h]+1,vertex_y[h+1]+1):
           Mceckerboard[i, j]=1
               
    phaseA = 0
    phaseB = np.pi
    phaseIn_0 = phaseA + np.zeros((slm_data_height, slm_data_width))
    phaseIn = a_incenter_b(phaseB * Mceckerboard, phaseIn_0)
    
    # img=plt.imshow(Mceckerboard)#cmap='Blues'
    # plt.axis('off')
    # plt.colorbar(img)
    # plt.show() 
    return mask_type, phaseIn



def checkerboard_2A(slm_data_height, slm_data_width):
    # checkerboard (define square single field size (px) of checkerboard with respect to slm width, checkerboard centered in the slm screen )
    mask_type = "input_mask_checkboard_2A"
    nsingleV = 2  ##number of fields single type, vertical direction
    nsingleH = 4  ##number of fields single type, horizontal direction
    # n=nsingleV*2 #number of fields both type
    # npixelsingle=np.int16(np.floor(dataHeight/n))
    n = nsingleH * 2  # number of fields both type
    npixelsingle = np.int16(np.floor(slm_data_width / n))
    Mceckerboard = np.kron([[1, 0] * nsingleH, [0, 1] * nsingleH] * nsingleV, np.ones((npixelsingle, npixelsingle)))
    phaseA = 0
    phaseB = np.pi
    phaseIn_0 = phaseA + np.zeros((slm_data_height, slm_data_width))
    phaseIn = a_incenter_b(phaseB * Mceckerboard, phaseIn_0)
    # plt.imshow(phaseIn)
    return mask_type, phaseIn

def checkerboard_2B(slm_data_height, slm_data_width):
    # checkerboard (define square single field size (px) of checkerboard with respect to slm width, checkerboard centered in the slm screen )
    mask_type = "input_mask_checkboard_2B"
    nsingleV = 2  ##number of fields single type, vertical direction
    nsingleH = 4  ##number of fields single type, horizontal direction
    # n=nsingleV*2 #number of fields both type
    # npixelsingle=np.int16(np.floor(dataHeight/n))
    n = nsingleH * 2  # number of fields both type
    npixelsingle = np.int16(np.floor(slm_data_width / n))
    Mceckerboard = np.kron([[1, 0] * nsingleH, [0, 1] * nsingleH] * nsingleV, np.ones((npixelsingle, npixelsingle)))
    phaseA = 0
    phaseB = np.pi
    phaseIn_0 = phaseB + np.zeros((slm_data_height, slm_data_width))
    phaseIn = a_incenter_b(phaseB *(-1)*(Mceckerboard-1), phaseIn_0)  
    # plt.imshow(phaseIn)
    return mask_type, phaseIn





def checkerboard_3A(slm_data_height, slm_data_width):
    # checkerboard (define square single field size (px) of checkerboard with respect to slm width, checkerboard centered in the slm screen )
    mask_type = "input_mask_checkboard_3A"
    nsingleV = 3  ##number of fields single type, vertical direction
    nsingleH = 6  ##number of fields single type, horizontal direction
    # n=nsingleV*2 #number of fields both type
    # npixelsingle=np.int16(np.floor(dataHeight/n))
    n = nsingleH * 2  # number of fields both type
    npixelsingle = np.int16(np.floor(slm_data_width / n))
    Mceckerboard = np.kron([[1, 0] * nsingleH, [0, 1] * nsingleH] * nsingleV, np.ones((npixelsingle, npixelsingle)))
    phaseA = 0
    phaseB = np.pi
    phaseIn_0 = phaseA + np.zeros((slm_data_height, slm_data_width))
    phaseIn = a_incenter_b(phaseB * Mceckerboard, phaseIn_0)
    # plt.imshow(phaseIn)
    return mask_type, phaseIn

def checkerboard_3B(slm_data_height, slm_data_width):
    # checkerboard (define square single field size (px) of checkerboard with respect to slm width, checkerboard centered in the slm screen )
    mask_type = "input_mask_checkboard_3B"
    nsingleV = 3  ##number of fields single type, vertical direction
    nsingleH = 6  ##number of fields single type, horizontal direction
    # n=nsingleV*2 #number of fields both type
    # npixelsingle=np.int16(np.floor(dataHeight/n))
    n = nsingleH * 2  # number of fields both type
    npixelsingle = np.int16(np.floor(slm_data_width / n))
    Mceckerboard = np.kron([[1, 0] * nsingleH, [0, 1] * nsingleH] * nsingleV, np.ones((npixelsingle, npixelsingle)))
    phaseA = 0
    phaseB = np.pi
    phaseIn_0 = phaseB + np.zeros((slm_data_height, slm_data_width))
    phaseIn = a_incenter_b(phaseB *(-1)*(Mceckerboard-1), phaseIn_0)  
    # plt.imshow(phaseIn)
    return mask_type, phaseIn





def checkerboard_4A(slm_data_height, slm_data_width):
    # checkerboard (define square single field size (px) of checkerboard with respect to slm width, checkerboard centered in the slm screen )
    mask_type = "input_mask_checkboard_4A"
    nsingleV = 4  ##number of fields single type, vertical direction
    nsingleH = 8  ##number of fields single type, horizontal direction
    # n=nsingleV*2 #number of fields both type
    # npixelsingle=np.int16(np.floor(dataHeight/n))
    n = nsingleH * 2  # number of fields both type
    npixelsingle = np.int16(np.floor(slm_data_width / n))
    Mceckerboard = np.kron([[1, 0] * nsingleH, [0, 1] * nsingleH] * nsingleV, np.ones((npixelsingle, npixelsingle)))
    phaseA = 0
    phaseB = np.pi
    phaseIn_0 = phaseA + np.zeros((slm_data_height, slm_data_width))
    phaseIn = a_incenter_b(phaseB * Mceckerboard, phaseIn_0)
    # plt.imshow(phaseIn)
    return mask_type, phaseIn

def checkerboard_4B(slm_data_height, slm_data_width):
    # checkerboard (define square single field size (px) of checkerboard with respect to slm width, checkerboard centered in the slm screen )
    mask_type = "input_mask_checkboard_4B"
    nsingleV = 4  ##number of fields single type, vertical direction
    nsingleH = 8  ##number of fields single type, horizontal direction
    # n=nsingleV*2 #number of fields both type
    # npixelsingle=np.int16(np.floor(dataHeight/n))
    n = nsingleH * 2  # number of fields both type
    npixelsingle = np.int16(np.floor(slm_data_width / n))
    Mceckerboard = np.kron([[1, 0] * nsingleH, [0, 1] * nsingleH] * nsingleV, np.ones((npixelsingle, npixelsingle)))
    phaseA = 0
    phaseB = np.pi
    phaseIn_0 = phaseB + np.zeros((slm_data_height, slm_data_width))
    phaseIn = a_incenter_b(phaseB *(-1)*(Mceckerboard-1), phaseIn_0)  
    # plt.imshow(phaseIn)
    return mask_type, phaseIn