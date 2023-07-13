    # %% SHOWING AND SAVING IMAGES
    # Before saving, let's make the necessary folders
    # in order to remember the input mask this lines are at the bottom of the file
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def make_run_folder(data_path, data_type_measure):
    today = datetime.now()
    data_day_dir = os.path.join(data_path, today.strftime('%Y_%m_%d'))
    data_measure_dir = os.path.join(data_day_dir, data_type_measure)
    # Make folder for the day
    if not os.path.exists(data_day_dir):
        print("Making folder for the day:", data_day_dir)
        os.makedirs(data_day_dir)
    # Make folder for the type of measure
    if not os.path.exists(data_measure_dir):
        print("Making folder for the type of measure:", data_measure_dir)
        os.makedirs(data_measure_dir)
    # Make folder for the run
    for ii in range(1, 9999):
        run_dir = os.path.join(data_measure_dir + f"/M{ii:05d}")
        if not os.path.exists(run_dir):
            print("Making folder for the current run:", run_dir)
            os.mkdir(run_dir)
            return run_dir
    return run_dir


def save_measures(run_dir, mask_type, Mframes_reference, Mframes,
                  phase_in_reference, phase_in, phase_shift, Nshifts):
    # First the folder for the input_mask
    mask_dir = os.path.join(run_dir, mask_type)
    if not os.path.exists(mask_dir):
        print("Making folder for the input mask:", mask_dir)
        os.makedirs(mask_dir)
    data_dir = mask_dir

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
        plt.close(fig1)

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
        plt.close(fig1)

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
    np.save(file_path, phase_in)
    file_path = os.path.join(pfile_save_dir, 'phasein_reference.npy')
    print("Saving:", file_path)
    np.save(file_path, phase_in_reference)

    # save input phase shift
    file_path = os.path.join(pfile_save_dir, 'phaseshifts.npy')
    print("Saving:", file_path)
    np.save(file_path, phase_shift)

    print("Done saving")
    return
