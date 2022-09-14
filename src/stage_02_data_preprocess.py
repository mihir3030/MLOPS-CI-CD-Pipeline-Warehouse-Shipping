import os
import numpy as np
import pandas as pd
import sklearn
import logging
import argparse
from src.utils.common import read_yaml, create_directory
from sklearn.preprocessing import LabelEncoder
import joblib

STAGE_NAME = "stage-02-data-preprocessing"

logging.basicConfig(
    filename = os.path.join("logs", "running_logs.log"),
    level = logging.INFO,
    format = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode = "a"
)


def data_preprocess(config_path):
    # reading two files
    config = read_yaml(config_path)

    # retrive dataset file path for config.yaml file
    artifacts_dir = config['artifacts']['artifacts_dir']
    raw_local_dir = config['artifacts']['raw_local_dir']
    raw_local_file = config['artifacts']['raw_local_file']

    # reading dataset from path
    raw_local_file_path = os.path.join(artifacts_dir, raw_local_dir, raw_local_file)
    df = pd.read_csv(raw_local_file_path)
  
    
    # applying DataPreprocessing steps
    # convert categorical columns into numeric for this we use LabelEncoder technique
    try:
        label_encoder = LabelEncoder()
        df = df.apply(label_encoder.fit_transform)
        logging.info("Categorical data successfully transform to numerical data")
    except Exception as e:
        logging.exception(e)
        raise e

    """ 
    Warehouseblock -- D=3, F=4, A=0, B=1, C=2
    Mode_of SHipment -- Flight=0, Ship=2, Air=1
    Product_importance -- Low=1, Medium=2, High=0
    Gender -- F=0, M=1
    
    """
    # save our new Dataset to Artifacts
    # read new dataset path from config
    final_data_dir = config['artifacts']['final_data_dir']
    final_data_file = config['artifacts']['final_data_file']
    final_data_dir_path = os.path.join(artifacts_dir, final_data_dir)

    # create new directory for final_data_dir
    create_directory([final_data_dir_path])

    final_data_file_path = os.path.join(final_data_dir_path, final_data_file)

    # save final dataset to our final data directory
    df.to_csv(final_data_file_path, index=False)
    logging.info(f"our preprocess final_dataset save successfully at {final_data_file_path}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # reading config file and parameters file 
    # config for all configurations and file path
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n\n*************************************************************************************************")
        logging.info(f">>>>>>>> {STAGE_NAME} started <<<<<<<< ")
        #load config.yaml file
        data_preprocess(config_path = parsed_args.config)
        logging.info(f">>>>>>>>>>> {STAGE_NAME} compleated <<<<<<<<<")
    
    except Exception as e:
        logging.exception(e)
        raise e