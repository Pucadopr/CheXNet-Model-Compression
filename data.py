"""
    Python script used for downloading all the datasets used in this
    project. 
    Data are downloaded to root folder

    python data.py --dataset='RSNA'
"""

import argparse
import academictorrents as at




parser = argparse.ArgumentParser()

# The dataset downloaded are the resized ones (224x224)
parser.add_argument('--dataset', default='NIH',
 choices=['NIH', 'RSNA', 'Pad-Chest', 'Openi'], 
 help='dataset name')
## Add more parser as need requires
args = parser.parse_args()

# dictionary of hash to download data from academictorrents
data_hash_dict = {'NIH': "e615d3aebce373f1dc8bd9d11064da55bdadede0", 
'RSNA': "95588a735c9ae4d123f3ca408e56570409bcf2a9", 
'Pad-Chest': "96ebb4f92b85929eadfb16761f310a6d04105797",
'Openi': "5a3a439df24931f410fac269b87b050203d9467d"}


# Print the dataset that its downloading
data_path = at.get(data_hash_dict.get(args.dataset)) # Download mnist dataset
print(data_path)


# Uncomment if the downloaded is a tar file (.tar)
# import tarfile

# files = tarfile.open("/file.tar") # specify the path to the tar file
# files.extractall("Data") # Specify the directory to extract too
# files.close()

# Uncomment if the downloaded is a zip file (.zip)
# import zipfile

# with zipfile.ZipFile("/path_file.zip", 'r') as zip_file:
#     zip_file.extractall("Data")


