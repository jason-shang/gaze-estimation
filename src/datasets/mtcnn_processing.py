import os
import json
import shutil
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, Process
import matplotlib.pyplot as plt

import mtcnn
from mtcnn.mtcnn import MTCNN

'''
Run MTCNN to extract face and eye info from app frames, and output in GazeCapture's format

Example usage: run `python mtcnn_processing.py --dir ../../data/app_data --out_dir ../../data/processed_data --threads 1` in terminal
'''

parser = argparse.ArgumentParser(description='Extract face & eye info from app frames')
parser.add_argument('--dir', help='Path to directory of session data outputted by app')
parser.add_argument('--out_dir', help='Path to new dataset following GazeCapture format')
parser.add_argument('--threads', default=1, help='Number of threads', type=int)

# dirs -> list of session directories
def extract_data(dirs, out_root): 
    detector = MTCNN()
    # process each directory (each session)
    for directory in tqdm(dirs):
        outdir = out_root + '/' + directory.split('/')[-2]
        with open(directory+'info.json') as info: 
            data = json.load(info)
            numFrames = data['totalFrames']
        
        # get all frames in current session directory
        jpeg_files = []
        for i in range(numFrames): 
            jpeg_files.append(directory + "frame_{}.jpg".format(i))
        
        # run MTCNN on each frame, append results accordingly
        face_x = []
        face_y = []
        face_w = []
        face_h = []
        face_valid = []

        l_eye_x = []
        l_eye_y = []
        l_eye_w = []
        l_eye_h = []
        l_eye_valid = []

        r_eye_x = []
        r_eye_y = []
        r_eye_w = []
        r_eye_h = []
        r_eye_valid = []

        frames = []

        # NOTE: these data should be outputted by app! skipped bc of sync issues
        screen_h = []
        screen_w = []
        orientation = []

        for i in range(len(jpeg_files)): 
            frame = jpeg_files[i]
            img = plt.imread(frame)
            detection_results = detector.detect_faces(img)

            # NOTE: these data should be outputted by app! 
            screen_h.append(390)
            screen_w.append(844)
            orientation.append(1) # assume portrait mode 

            if detection_results == []:
                face_valid.append(0)

                # TODO: implement additional checks for eye detection validity
                l_eye_valid.append(0)
                r_eye_valid.append(0)

                face_x.append(0)
                face_y.append(0)
                face_w.append(0)
                face_h.append(0)
                
                l_eye_x.append(0)
                l_eye_y.append(0)
                l_eye_w.append(0)
                l_eye_h.append(0)

                r_eye_x.append(0)
                r_eye_y.append(0)
                r_eye_w.append(0)
                r_eye_h.append(0)
            else: 
                # assume only one person
                detection_result = detection_results[0]

                face_valid.append(1)
                l_eye_valid.append(1)
                r_eye_valid.append(1)

                face_x.append(detection_result['box'][0])
                face_y.append(detection_result['box'][1])
                face_w.append(detection_result['box'][2])
                face_h.append(detection_result['box'][3])

                l_eye_box, r_eye_box = getEyeBoundingBoxes(detection_result)
                
                l_eye_x.append(l_eye_box[0])
                l_eye_y.append(l_eye_box[1])
                l_eye_w.append(l_eye_box[2])
                l_eye_h.append(l_eye_box[3])

                r_eye_x.append(r_eye_box[0])
                r_eye_y.append(r_eye_box[1])
                r_eye_w.append(r_eye_box[2])
                r_eye_h.append(r_eye_box[3])

            # copy each frame over to output directory
            print(frame)
            fname = str(i).zfill(5)
            print("fname: ", fname)
            frames.append(fname + '.jpg')

            frames_directory = outdir+'/frames/'
            if not os.path.exists(frames_directory):
                os.makedirs(frames_directory)
            shutil.copy(frame, frames_directory+fname+'.jpg')

        face_data = {}
        face_data['X'] = face_x
        face_data['Y'] = face_y
        face_data['W'] = face_w
        face_data['H'] = face_h
        face_data['IsValid'] = face_valid

        l_eye_data = {}
        l_eye_data['X'] = l_eye_x
        l_eye_data['Y'] = l_eye_y
        l_eye_data['W'] = l_eye_w
        l_eye_data['H'] = l_eye_h
        l_eye_data['IsValid'] = l_eye_valid

        r_eye_data = {}
        r_eye_data['X'] = r_eye_x
        r_eye_data['Y'] = r_eye_y
        r_eye_data['W'] = r_eye_w
        r_eye_data['H'] = r_eye_h
        r_eye_data['IsValid'] = r_eye_valid

        screen_data = {}
        screen_data['H'] = screen_h
        screen_data['W'] = screen_w
        screen_data['Orientation'] = orientation

        # copy over info.json & screen.json
        shutil.copy(directory+'info.json', outdir+'/info.json')
        shutil.copy(directory+'screen.json', outdir+'/screen.json')

        frames_file = outdir+'/frames.json'
        with open(frames_file, 'w') as outfile:
            json.dump(frames, outfile)

        face_file = outdir+'/face.json'
        with open(face_file, 'w') as outfile:
            json.dump(face_data, outfile)
        
        l_eye_file = outdir+'/l_eye.json'
        with open(l_eye_file, 'w') as outfile:
            json.dump(l_eye_data, outfile)

        r_eye_file = outdir+'/r_eye.json'
        with open(r_eye_file, 'w') as outfile:
            json.dump(r_eye_data, outfile)

        # screen_file = outdir+'/screen.json'
        # with open(screen_file, 'w') as outfile:
        #     json.dump(screen_data, outfile)

def getEyeBoundingBoxes(detection_result): 
    _, _, face_w, face_h = detection_result['box']
    l_eye = detection_result['keypoints']['left_eye']
    r_eye = detection_result['keypoints']['right_eye']
    
    # could determine proportions empirically (average of all proportions in GazeCapture dataset?)
    width_proportion = height_proportion = 4
    eye_w, eye_h = face_w / width_proportion, face_h / height_proportion

    l_eye_box = (
        int(l_eye[0] - (eye_w / 2)),
        int(l_eye[1] - (eye_h / 2)), 
        eye_w, 
        eye_h 
    )

    r_eye_box = (
        int(r_eye[0] - (eye_w / 2)), 
        int(r_eye[1] - (eye_h / 2)), 
        eye_w, 
        eye_h
    )

    return l_eye_box, r_eye_box

def assign_work(path, out_dir, threads):
    procs = []
    # get all session directories
    files = glob(path+"/*/")
    chunk = len(files)//threads
    print(len(files))
    for i in range(threads): 
        f = files[i*chunk:(i+1)*chunk]
        if(i==threads-1):
            f = files[i*chunk:]
        
        proc = Process(target=extract_data, args=(f, out_dir))
        procs.append(proc)
        proc.start()
        
    for proc in procs:
        proc.join()

def main():
    args = parser.parse_args()
    assign_work(args.dir, args.out_dir, args.threads)
    print("Processing Complete")

if __name__=="__main__":
    main()
