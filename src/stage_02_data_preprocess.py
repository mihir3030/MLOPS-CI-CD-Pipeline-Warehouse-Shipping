import os
import pandas as pd
import sklearn
import logging
import argparse
from src.utils.common import read_yaml, create_directory

STAGE = "stage-02-data-preprocessing"

logging.basicConfig(
    filename = os.path.join("logs", "running_logs.log"),
    level = logging.INFO,
    format = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode = "a"
)


def data_preprocess(config_path):
    config = read_yaml(config_path)
    



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    data_preprocess(config_path = parsed_args.config)