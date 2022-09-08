import os
import argparse
import logging
from src.utils.common import read_yaml, create_directory
import pandas as pd

# use current stage name
STAGE_NAME = "stage-01-load_data"

# create logging information - hos t osave logs
logging.basicConfig(
    filename = os.path.join("logs", "running_logs.log"),
    level = logging.INFO,
    format = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode = "a"
)

# main function for load and save dataset
def load_data(config_path):
    #read config yaml file
    config = read_yaml(config_path) 

    # read remote data from cloud or local
    remote_data_path = config['remote_data_path'] 
    df = pd.read_csv(remote_data_path) # load remote data into pandas DataFrame
    
    # read our artifacts directories path from config.yaml file
    artifacts_dir = config['artifacts']['artifacts_dir']
    raw_local_dir = config['artifacts']['raw_local_dir']

    # join directory to save out dataset
    raw_local_dir_path = os.path.join(artifacts_dir, raw_local_dir)

    # creat directory function from utils file to create our artifacts and dataset directory
    create_directory([raw_local_dir_path])

    # get our file location to save our dataset into artifacts
    raw_local_file = config['artifacts']['raw_local_file']
    raw_local_file_path = os.path.join(raw_local_dir_path, raw_local_file)

    try:
        # save our dataset into our artifacts/raw_local_dir/data.csv
        df.to_csv(raw_local_file_path, index=False)
        logging.info(f"dataset save at {raw_local_file_path}")
    except Exception as e:
        logging.exception(e)
        raise e

    


if __name__ == "__main__":
    # create argument parser to give path of config.yaml file
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()
    
    try:
        logging.info("\n\n*************************************************************************************************")
        logging.info(f">>>>>>>> {STAGE_NAME} started <<<<<<<< ")
        #load config.yaml file
        load_data(config_path=parsed_args.config)
        logging.info(f">>>>>>>>>>> {STAGE_NAME} compleated <<<<<<<<<")
    
    except Exception as e:
        logging.exception(e)
        raise e