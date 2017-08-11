import os
import dicom
import shutil


def classify_dicom_img(dicom_folder_name: str, output_folder_name: str) -> None:
    """
    分析dicom文件，对其中的CT文件和PET文件
    进行分类并排序(根据SeriesNumber和InstanceNumber)，
    最终输出到输出文件夹下
    :param dicom_folder_name: dicom文件所在文件夹(绝对路径)
    :param output_folder_name：输出文件的文件夹(绝对路径)
    :return: None
    """
    # 得到*dicom_folder_name*文件夹下所有的文件（绝对路径，不包括子文件夹里的文件）
    file_names = [os.path.join(dicom_folder_name, x) for x in os.listdir(dicom_folder_name)]
    file_names = filter(lambda x: os.path.isfile(x), file_names)
    for file_name in file_names:
        dicom_data = dicom.read_file(file_name)

        # 根据SeriesNumber在output_folder_name下创建子文件夹(如果不存在此文件夹)
        temp_folder_name = os.path.join(output_folder_name, str(dicom_data.get("SeriesNumber")))
        if not (os.path.exists(temp_folder_name) and os.path.isdir(temp_folder_name)):
            os.mkdir(temp_folder_name)

        # 根据"{Modality}_{InstanceNumber}"的格式，在先前的子文件夹下创建文件(如果不存在此文件)
        # 如果文件已经存在，重命名为***_2, ***_3...
        temp_file_name = "{}_{}".format(str(dicom_data.get("Modality")), str(dicom_data.get("InstanceNumber")))
        temp_file_name = os.path.join(temp_folder_name, temp_file_name)

        i = 0
        while True:
            if not (os.path.exists(temp_file_name) and os.path.isfile(temp_file_name)):
                shutil.copy(file_name, temp_file_name)
                break
            i += 1
            temp_file_name = "{}_{}".format(temp_file_name[:-2], i)


if __name__ == '__main__':
    # classify_dicom_img.classify_dicom_img(r"F:\PT06535\DICOMIMG", r"F:\PT06535")
    pass
