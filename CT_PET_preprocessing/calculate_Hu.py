import numpy as np


def calculate_hu(dicom_data) -> np.ndarray:
    return dicom_data.pixel_array.copy() * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
