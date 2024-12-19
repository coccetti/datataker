# Functions to take pictures


def acquisition(slm, camera, mask_type, phase_shift, Nshifts, slm_data_width, slm_data_height,
                phase_in, Mframes):
    # Take pictures of phaseIn_reference and of phaseIn
    import holoeye
    from holoeye import slmdisplaysdk
    # from pypylon import pylon
    # from pypylon import genicam

    # laser_wavelength_nm = 532.0
    # camera_exposure_time = 50  # ms
# 
    # # Camera
    # camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    # camera.Open()
    # # enable all chunks
    # camera.ChunkModeActive = True
    # for cf in camera.ChunkSelector.Symbolics:
    #     camera.ChunkSelector = cf
    #     camera.ChunkEnable = True
    # camera.ExposureTime.SetValue(camera_exposure_time)  # ms

    # # Initializes the SLM library
    # slm = slmdisplaysdk.SLMInstance()
    # error = slm.open()
    # assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
    # wavefrontfile = (
    #     r'C:\Users\LFC_01\Documents\SLM_PLUTO_MATERIAL\Wavefront_Correction_Function\U.14-2040-182427-2X-00-05_7020-1 6010-1086.h5')
    # error = slm.wavefrontcompensationLoad(wavefrontfile, laser_wavelength_nm,
    #                                       slmdisplaysdk.WavefrontcompensationFlags.NoFlag, 0, 0)
    # assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
    # error = slm.wavefrontcompensationLoad(wavefrontfile, laser_wavelength_nm,
    #                                       slmdisplaysdk.WavefrontcompensationFlags.NoFlag, 0, 0)
    # assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

    for i in range(Nshifts):
        print("Phase shift:", f"{phase_shift[i]:.4f}")

        # # Code for the reference
        # print("  Taking shot for reference")
        # phaseData = slmdisplaysdk.createFieldSingle(slm_data_width, slm_data_height) + phase_in_reference + phase_shift[i]
        # error = slm.showPhasevalues(phaseData)  # display phase values on the SLM
        # assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
        # result_reference = camera.GrabOne(100)  # grab frame file on the camera
        # Mframes_reference[:, :, i] = result_reference.Array  # extract numerical matrix and build 3D frame matrix

        # Code for the phaseIn selected
        print("  Taking shot for", mask_type)
        phaseData = slmdisplaysdk.createFieldSingle(slm_data_width, slm_data_height) + phase_in + phase_shift[i]
        error = slm.showPhasevalues(phaseData)  # display phase values on the SLM
        assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
        result = camera.GrabOne(100)  # grab frame file on the camera
        Mframes[:, :, i] = result.Array  # extract numerical matrix and build 3D frame matrix

    # camera.Close()
    # slm.close()
    return Mframes