import numpy as np
from datetime import timedelta
from math import pow


def calculate_suv(dicom_data, mode_str: str) -> np.ndarray:
    # get dicom data
    img_data = dicom_data.pixel_array.copy()
    rescale = dicom_data.RescaleSlope
    intercept = dicom_data.RescaleIntercept
    weight = dicom_data.PatientWeight
    height = 100 * dicom_data.PatientSize
    sex = dicom_data.PatientSex
    factor = 1000

    # calculate elapsed time
    start_time_str = dicom_data.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
    start_time = timedelta(hours=int(start_time_str[0:2]), minutes=int(start_time_str[2:4]),
                           seconds=int(start_time_str[4:6]))
    end_time_str = dicom_data.SeriesTime
    end_time = timedelta(hours=int(end_time_str[0:2]), minutes=int(end_time_str[2:4]), seconds=int(end_time_str[4:6]))
    elapsed_time = (end_time - start_time).total_seconds()

    # calculate actual_activity
    tracer_activity = dicom_data.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
    half_life = dicom_data.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
    actual_activity = tracer_activity * pow(2, -elapsed_time / half_life)

    if mode_str == "BW":
        return rescale * (img_data + intercept) * weight * factor / actual_activity
    elif mode_str == "LBM":
        if sex.upper() == 'F':
            wLBM = 1.07 * weight - 148 * (weight / height ** 2)
        else:
            wLBM = 1.10 * weight - 120 * (weight / height ** 2)
        return rescale * (img_data + intercept) * wLBM * factor / actual_activity
    elif mode_str == "BSA":
        bsa = (weight ** 0.425) * (height ** 0.725) * 0.007184
        return rescale * (img_data + intercept) * bsa * 10000 / actual_activity
