import os
import shutil


def get_color_dict():

    color_dict = dict()
    color_dict[0] = [255, 2, 4]
    color_dict[1] = [192, 52, 111]
    color_dict[2] = [235, 233, 36]
    color_dict[3] = [244, 126, 39]
    color_dict[4] = [235, 181, 71]
    color_dict[5] = [0, 173, 48]
    color_dict[6] = [0, 72, 24]
    color_dict[7] = [90, 30, 106]
    color_dict[8] = [113, 122, 50]
    color_dict[9] = [192, 173, 69]
    color_dict[10] = [214, 236, 101]
    color_dict[11] = [172, 183, 187]
    color_dict[12] = [39, 76, 212]
    color_dict[13] = [96, 229, 254]

    return color_dict


def create_folder(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)

    os.mkdir(folder_name)
