import os
import logging
import argparse
from sklearn.metrics import confusion_matrix, classification_report
from src.utils.common import read_yaml, create_directory

STAGE_NAME = 'stage_04_evaluation'

logging.basicConfig(
    filename = os.path.join("logs", "running_logs.log"),
    level = logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s:",
    filemode = "a"
)

def eval(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    print(config, params)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    eval(config_path=parsed_args.config, params_path=parsed_args.params)