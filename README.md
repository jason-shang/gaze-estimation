### File Structure
* `checkpoints/` contains the weight files for pre-trained models
* `data/` (uncommitted)
    * `data/app_data/` contains session data outputted from the app ([here](https://github.com/jason-shang/GazeTrackApp))
    * `data/converted_data/` contains the final converted data files after running the dataset conversion scripts (ready to use for model)
    * `data/processed_data/` contains frames along with their face, eye and session/devic data
* `notebooks/` contains all Jupyter notebooks (for data exploration and debugging)
* `src/`
    * `src/datasets` contains the Python scripts for converting from app data to processed data, then from processed data to converted data
    * `src/pred` contains code for the gaze estimation model
* `torchscript/` contains the torchscript models for on-device mobile inference