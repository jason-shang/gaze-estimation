### File Structure
* `checkpoints/` contains the weight files for pre-trained models
* `data/` (uncommitted)
    * `data/app_data/` contains session data outputted from the app ([here](https://github.com/jason-shang/GazeTrackApp))
    * `data/converted_data/` contains the final converted data files after running the dataset conversion scripts (ready to use for model)
    * `data/processed_data/` contains frames along with their face, eye and session/devic data
    * `data/results/` contains results of predictions in JSON format; each results JSON file contains: predicted gaze point for each frame in cm from camera, ground truth point for each frame in cm from camera, orientation of device for that frame, and error/Euclidean distance between gaze point and ground truth point for that frame
* `notebooks/` contains all Jupyter notebooks (for data exploration and debugging)
* `src/`
    * `src/datasets` contains the Python scripts for converting from app data to processed data, then from processed data to converted data
    * `src/pred` contains code for the gaze estimation model
* `torchscript/` contains the torchscript models for on-device mobile inference

### Install
This project uses Python 3. To install the required packages, run the following command: `pip install -r requirements.txt`. 

### Usage
After retrieving session data from app (assuming we put it in `data/app_data`), cd inside src/datasets, then run
```
# resize and flip image (run on all sessions in directory)
python app_data_processing.py --dir ../../data/app_data/ --out_dir ../../data/processed_data --threads 1

# convert to desirable format (run on all session in directory)
python data_converter.py --dir ../../data/processed_data --out_dir ../../data/converted_data --threads 1

# add eye key points (run on specific session)
python add_eye_kps.py --dir ../../data/converted_data/converted_041823-19:30

# predict
cd ../pred
python predict.py --dir ../../data/converted_data --out_dir ../../data/results
```