import json
import shutil
import numpy as np
from glob import glob
from multiprocessing import Pool, Process
import scipy.io

'''
Convert MPIIFaceGaze dataset to the one used in "Accelerating eye movement research via accurate and affordable smartphone eye tracking" and prepare data for easy use in PyTorch.
1. Keep only portrait orientation images (do we have orientation data?)
2. Keep only images that have valid eye detections
3. Split data of each participant into train, test, split. 
'''

import argparse
parser = argparse.ArgumentParser(description='Convert the MPIIFaceGaze dataset')
parser.add_argument('--dir', default="../../dataset/", help='Path to unzipped MPIIFaceGaze dataset')
parser.add_argument('--out_dir', default="../../dataset/", help='Path to new dataset should have train, val, test folders, each with image, meta folders')
parser.add_argument('--threads', default=1, help='Number of threads', type=int)

def convert_dataset(files, out_root):
    for i in tqdm(files): 
        screenSize = scipy.io.loadmat('./Calibration/screenSize.mat')
        # annotation = 
