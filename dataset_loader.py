import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from utils import check_file_exists, check_path_exists
from PIL import Image


class DatasetLoader(Dataset):

    def __init__(self, img_dir, transform=None, masks=False):

        self.transform = transform
        self.path_to_images = img_dir
        self.df = pd.read_csv(os.path.join(os.path.dirname(img_dir), "Data_Entry_2017.csv"))
        self.masks = pd.read_csv(os.path.join(os.path.dirname(img_dir), "BBox_List_2017.csv"), 
                names=["Image Index","Finding Label","x","y","w","h","_1","_2","_3"],
               skiprows=1)


        check_path_exists(self.path_to_images)
        check_file_exists(self.df)

        if masks:
            check_file_exists(self.masks)

        self.df = self.df.set_index("Image Index")

        self.diseases = [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia']
           

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.index[idx]))
        image = image.convert('RGB')

        label = np.zeros(len(self.diseases), dtype=int)

        for i in range(0, len(self.diseases)):  
            if(self.df[self.diseases[i].strip()].iloc[idx].astype('int') > 0):
                label[i] = self.df[self.diseases[i].strip()
                                   ].iloc[idx].astype('int')

        if self.transform:
            image = self.transform(image)

        return (image, label, self.df.index[idx])
