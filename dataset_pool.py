from my_utils.dataset import DataSet
import os

ROOT = "dataset"


# UBD_BOTANICAL = DataSet("UBD_Botanical", 45)
# VNPLANTS = DataSet("VNPlants", 200)
# FLAVIA = DataSet("Flavia", 32)
# MALAYA_KEW = DataSet("MalayaKew", 44)

def get_dataset(folder_name):
    trainfolder = os.path.join(ROOT, folder_name, 'train')
    dirlist_trian = os.listdir(trainfolder)
    num_class = len(dirlist_trian)
    return DataSet(folder_name, num_class)
