import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def rename(fnames, class_path):
    os.chdir(class_path)
    noof_data = len(fnames)
    for i in range(noof_data):
        os.rename(fnames[i], str(i)+".png")


test_data_path = "../dataset/inaturalist_12K/test/"
train_data_path = "../dataset/inaturalist_12K/train/"
classes = os.listdir(train_data_path)

print("Classes:")
print(classes)

# Rename all training data file names
dist = {}
current_path = os.getcwd()
for i in tqdm(classes):
    class_path = train_data_path + i + "/"
    fnames = os.listdir(class_path)
    dist[i] = len(fnames)

    rename(fnames, class_path)
    os.chdir(current_path)

# Rename all testing data file names
dist_test = {}
current_path = os.getcwd()
for i in tqdm(classes):
    class_path = test_data_path + i + "/"
    fnames = os.listdir(class_path)
    dist_test[i] = len(fnames)

    rename(fnames, class_path)
    os.chdir(current_path)
