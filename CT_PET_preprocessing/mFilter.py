def m_filter(arr, threshold):
    return arr * (1 - (arr < threshold))



if __name__ == '__main__':
    import dicom
    import numpy as np
    from scipy.io import loadmat
    from CT_PET_preprocessing.compare_matrix import compare_2D_matrix
    from CT_PET_preprocessing.calculate_Hu import calculate_hu
    from CT_PET_preprocessing.calculate_SUV import calculate_suv

    arr = calculate_suv(dicom.read_file("C:\\PT06535\\8\\PT_1"), "LBM")
    arr2 = calculate_hu(dicom.read_file("C:\\PT06535\\7\\CT_1"))
    arr = m_filter(arr, 2.0)
    arr2 = m_filter(arr2, 150)
    arr = arr.transpose([1, 0])
    arr2 = arr2.transpose([1, 0])
    print(arr.dtype, arr.shape)
    print(arr2.dtype, arr2.shape)


    arr3 = loadmat("test.mat")["test_imgfs_1"][0]
    arr4 = loadmat("test.mat")["test_imgfh_1"][0].astype(np.float64)
    print(arr3.dtype, arr3.shape)
    print(arr4.dtype, arr4.shape)

    compare_2D_matrix(arr, arr3, 100, 0)
    compare_2D_matrix(arr2, arr4, 100, 0)

