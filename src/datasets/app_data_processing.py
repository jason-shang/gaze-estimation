import os
import json
import shutil
from glob import glob
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
from multiprocessing import Pool, Process

parser = argparse.ArgumentParser(description='Prepare app frames for further processing')
parser.add_argument('--dir', help='Path to directory of session data outputted by app')
parser.add_argument('--out_dir', help='Path to directory that stores processed data')
parser.add_argument('--threads', default=1, help='Number of threads', type=int)

'''
1. Resize images to device dimensions (app only crops to proper aspect ratio)
2. Mirror images 

Example usage: run `python app_data_processing.py --dir ../../data/app_data --out_dir ../../data/processed_data --threads 1` in terminal
'''

def resize_image(img, screenW, screenH): 
    image = Image.fromarray(img)
    return image.resize((screenW, screenH))

def process_data(dirs, out_root): 
    # process each directory (each session)
    for directory in tqdm(dirs): 
        outdir = out_root + '/' + directory.split('/')[-2]

        # copy over all json files
        json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
        for f in json_files:
            outdir_w_frames = outdir + '/frames'
            if not os.path.exists(outdir_w_frames):
                os.makedirs(outdir_w_frames)
            shutil.copy(directory + '/' + f, outdir+'/'+f)

        screenInfo = json.load(open(directory + 'screen.json'))
        screenHs = screenInfo['H']
        screenWs = screenInfo['W']

        # resize and flip each frame
        numFrames = len(os.listdir(directory+'frames'))
        for i in range(numFrames):
            fname = str(i).zfill(5) + '.jpg'
            img_read = plt.imread(directory+'frames/'+fname)
            img = resize_image(img_read, screenWs[i], screenHs[i])
            flipped_img = np.fliplr(img)

            # save the new JPEG file
            output_path = outdir + '/frames/' + fname
            plt.imsave(output_path, flipped_img)

def assign_work(path, out_dir, threads):
    procs = []
    files = glob(path+"/*/")
    chunk = len(files)//threads
    for i in range(threads): 
        f = files[i*chunk:(i+1)*chunk]
        if(i==threads-1):
            f = files[i*chunk:]
        
        proc = Process(target=process_data, args=(f, out_dir))
        procs.append(proc)
        proc.start()
        
    for proc in procs:
        proc.join()

def main():
    args = parser.parse_args()
    assign_work(args.dir, args.out_dir, args.threads)

if __name__=="__main__":
    main()