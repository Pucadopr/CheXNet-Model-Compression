import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
import pprint
from utils import check_file_exists, check_path_exists
from PIL import Image
from skimage.io import imread, imsave
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os,sys,os.path
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random


class NIHDatasetLoader(Dataset):

    def __init__(self, img_dir, xray_csv, bbox_csv, transform=None, masks=False):

        self.transform = transform
        self.path_to_images = img_dir
        self.df = pd.read_csv(xray_csv)
        self.masks = pd.read_csv((bbox_csv), 
                names=["Image Index","Finding Label","x","y","w","h","_1","_2","_3"],
               skiprows=1)


        check_path_exists(self.path_to_images)
        check_file_exists(xray_csv)

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
            'Hernia',
            'Enlarged_Cardiomediastinum',
            'Lung_Lesion',
            'Fracture',
            'Lung_Opacity']
           

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

        return (image, label)

class NIHDataset(Dataset):
    """
    NIH ChestX-ray8 dataset

    Dataset release website:
    https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community
    
    Download full size images here:
    https://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a
    
    Download resized (224x224) images here:
    https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
    """
    def __init__(self, imgpath, 
                 csvpath, 
                 bbbox_path,
                 views=["PA"],
                 transform=None, 
                 data_aug=None, 
                 nrows=None, 
                 seed=0,
                 pure_labels=False, 
                 unique_patients=True,
                 normalize=True,
                 pathology_masks=False):
        
        super(NIHDataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.csv = pd.read_csv(csvpath, nrows=nrows)
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks
        
        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia", 
                            "Enlarged_Cardiomediastinum","Lung_Lesion","Fracture","Lung_Opacity"]                            
        
        self.pathologies = sorted(self.pathologies)
        
        self.normalize = normalize
        # Load data
        # self.check_paths_exist()
        self.MAXVAL = 255  # Range [0 255]
        
        if type(views) is not list:
            views = [views]
        self.views = views
        # Remove images with view position other than specified
        self.csv = self.csv[self.csv['View Position'].isin(self.views)]
        
        # Remove multi-finding images.
        if pure_labels:
            self.csv = self.csv[~self.csv["Finding Labels"].str.contains("\|")]
        
        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first()
        
        self.csv = self.csv.reset_index()
        
        ####### pathology masks ########
        # load nih pathology masks
        self.pathology_maskscsv = pd.read_csv(bbbox_path, 
                names=["Image Index","Finding Label","x","y","w","h","_1","_2","_3"],
               skiprows=1)
        
        # change label name to match
        self.pathology_maskscsv["Finding Label"][self.pathology_maskscsv["Finding Label"] == "Infiltrate"] = "Infiltration"
        self.csv["has_masks"] = self.csv["Image Index"].isin(self.pathology_maskscsv["Image Index"])
        ####### pathology masks ########    
            
        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(self.csv["Finding Labels"].str.contains(pathology).values)
            
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={} views={}".format(len(self), self.views)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        
        
        imgid = self.csv['Image Index'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        #print(img_path)
        img = imread(img_path)
        if self.normalize:
            img = normalize(img, self.MAXVAL)  

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        sample["img"] = img[None, :, :]                    

        transform_seed = np.random.randint(2147483647)
        
        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape[2])
            
        if self.transform is not None:
            random.seed(transform_seed)
            sample["img"] = self.transform(sample["img"])
            if self.pathology_masks:
                for i in sample["pathology_masks"].keys():
                    random.seed(transform_seed)
                    sample["pathology_masks"][i] = self.transform(sample["pathology_masks"][i])
  
        if self.data_aug is not None:
            random.seed(transform_seed)
            sample["img"] = self.data_aug(sample["img"])
            if self.pathology_masks:
                for i in sample["pathology_masks"].keys():
                    random.seed(transform_seed)
                    sample["pathology_masks"][i] = self.data_aug(sample["pathology_masks"][i])
            
        return sample
    
    def get_mask_dict(self, image_name, this_size):

        base_size = 1024
        scale = this_size/base_size

        images_with_masks = self.pathology_maskscsv[self.pathology_maskscsv["Image Index"] == image_name]
        path_mask = {}

        for i in range(len(images_with_masks)):
            row = images_with_masks.iloc[i]
            
            # don't add masks for labels we don't have
            if row["Finding Label"] in self.pathologies:
                mask = np.zeros([this_size,this_size])
                xywh = np.asarray([row.x,row.y,row.w,row.h])
                xywh = xywh*scale
                xywh = xywh.astype(int)
                mask[xywh[1]:xywh[1]+xywh[3],xywh[0]:xywh[0]+xywh[2]] = 1
                
                # resize so image resizing works
                mask = mask[None, :, :] 
                
                path_mask[self.pathologies.index(row["Finding Label"])] = mask
        
        return path_mask

def normalize(sample, maxval):
    """Scales images to be roughly [-1024 1024]."""
    sample = (2 * (sample.astype(np.float32) / maxval) - 1.) * 1024
    #sample = sample / np.std(sample)
    return sample