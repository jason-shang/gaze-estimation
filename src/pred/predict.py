import argparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool, Process

import sys
sys.path.insert(0, '../datasets/')
sys.path.insert(1, '../pred/')
from model import gazetrack_model
from gazetrack_data import gazetrack_dataset
import numpy as np

'''
Run model and get predictions, along with ground truth labels, for each session

Sample usage: `python predict.py --dir ../../data/converted_data --out_dir ../../data/results
'''

parser = argparse.ArgumentParser(description='Predict gaze point, output predictions & truth labels')
parser.add_argument('--dir', help='Path to directory of converted session data')
parser.add_argument('--out_dir', help='Path to directory which will contain prediction & truth labels for each session')
parser.add_argument('--weight_file', default='../../checkpoints/checkpoint.ckpt', help='Path to model weight file')
parser.add_argument('--threads', default=1, help='Number of threads', type=int)

def euc(a, b):
    return np.sqrt(np.sum(np.square(a - b), axis=1))

def predict(dirs, out_root, weight_file): 
    for directory in tqdm(dirs): 
        f = directory + 'train/images/'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = gazetrack_model()

        w = torch.load(weight_file, map_location=device)['state_dict']
        model.load_state_dict(w)
        model.to(device)
        model.eval()

        preds, gt, ors = [], [], []
        test_dataset = gazetrack_dataset(f, phase='test')
        test_dataloader = DataLoader(test_dataset, batch_size=512, num_workers=10, pin_memory=False, shuffle=False,)

        for j in tqdm(test_dataloader):
            leye, reye, kps, target, ori = j[1].to(device), j[2].to(device), j[3].to(device), j[4].to(device), j[-1]
            
            with torch.no_grad():
                pred = model(leye, reye, kps)
            pred = pred.detach().cpu().numpy()
            preds.extend(pred)
            ors.extend(ori)
            
            gt.extend(target.detach().cpu().numpy())
        
        preds = np.array(preds)
        gt = np.array(gt)
        ors = np.array(ors)
        dist = euc(preds, gt)

        results = {}
        results['preds'] = preds.tolist()
        results['ground_truths'] = gt.tolist()
        results['orientation'] = ors.tolist()
        results['error'] = dist.tolist()

        path = '/' + out_root + '/' + directory + '_results.json'
        with open(path, 'w') as outfile: 
            json.dump(results, outfile)

def assign_work(path, out_dir, threads, weight_file):
    procs = []
    # get all session directories
    files = glob(path+"/*/")
    chunk = len(files)//threads
    print(len(files))
    for i in range(threads): 
        f = files[i*chunk:(i+1)*chunk]
        if(i==threads-1):
            f = files[i*chunk:]
        
        proc = Process(target=predict, args=(f, out_dir, weight_file))
        procs.append(proc)
        proc.start()
        
    for proc in procs:
        proc.join()

def main():
    args = parser.parse_args()
    assign_work(args.dir, args.out_dir, args.threads, args.weight_file)
    print("Processing Complete")

if __name__=="__main__":
    main()