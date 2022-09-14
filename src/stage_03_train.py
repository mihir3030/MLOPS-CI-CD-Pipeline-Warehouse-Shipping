import os
import numpy as np
import pandas as pd
import argparse
import logging
from src.utils.common import read_yaml, create_directory
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
import joblib



STAGE_NAME = "stage_03_train"

logging.basicConfig(
    filename = os.path.join('logs', 'running_logs.log'),
    level = logging.INFO,
    format = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s:",
    filemode='a'
)

def train(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    # read dataset path from config.yaml file
    artifacts_dir = config['artifacts']['artifacts_dir']
    final_data_dir = config['artifacts']['final_data_dir']
    final_data_file = config['artifacts']['final_data_file']
    final_data_file_path = os.path.join(artifacts_dir, final_data_dir, final_data_file)
    logging.info("read config.yaml file successfully")

    # load dataset into pandas 
    df = pd.read_csv(final_data_file_path)
    
    # read parameters for train_test_split from params.yaml file
    test_size = params['base']['test_size']
    random_state = params['base']['random_state']

    # split data into train and test
    x = df.drop(['Reached.on.Time_Y.N'], axis=1)
    y = df['Reached.on.Time_Y.N']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    
    # reading algorithms parameter from params.yaml file
    dt_max_depth = params['base_model']['DecisionTree']['max_depth']
    dt_min_samples_leaf = params['base_model']['DecisionTree']['min_samples_leaf']

    ab_lr = params['base_model']['AdaBoost']['learning_rate']
    ab_n_estimators = params['base_model']['AdaBoost']['n_estimators']

    gb_lr = params['base_model']['GradientBoosting']['learning_rate']
    gb_n_estimators = params['base_model']['GradientBoosting']['n_estimators']
    logging.info("parameters load successfully from params.yaml file")

    cross_validation = params['base_model']['cross_validation']
    # we have our train and test dataset
    # stacking model stacking model uses multiple algorithms
    # define base model 
    estimators = [
        ('CART', DecisionTreeClassifier(max_depth=dt_max_depth, min_samples_leaf=dt_min_samples_leaf)),
        ('AB', AdaBoostClassifier(learning_rate=ab_lr, n_estimators=ab_n_estimators)),
        ('GB', GradientBoostingClassifier(learning_rate=gb_lr, n_estimators=gb_n_estimators))
    ]

    # define final model
    final_estimator = AdaBoostClassifier(learning_rate=ab_lr, n_estimators=ab_n_estimators)
    # now join base estimators and final nestimator using stacking 
    model =StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=cross_validation) 

    # start training
    model.fit(x_train, y_train)
    logging.info("model trained sucessfully")
    
    # saving model to artifacts 
    # first we need to gather model_dir path from config.yaml file
    model_dir = config['artifacts']['model_dir']
    model_dir_path = os.path.join(artifacts_dir, model_dir)

    # create directory for model
    create_directory([model_dir_path])

    # model_name and model_path
    model_name = config['artifacts']['model_name']
    model_name_path = os.path.join(model_dir_path, model_name)

    # save model
    joblib.dump(model, model_name_path)
    logging.info(f"model save succefully at {model_name_path}")



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # reading config file and parameters file 
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    try:
        logging.info("\n\n*************************************************************************************************")
        logging.info(f">>>>>>>> {STAGE_NAME} started <<<<<<<< ")
        #load config.yaml file
        train(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>>>>>>>> {STAGE_NAME} compleated <<<<<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e
