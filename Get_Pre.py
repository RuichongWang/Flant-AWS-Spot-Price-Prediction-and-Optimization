import json
from Preprocess import *
from Model import *

def load_train_config():
    with open('../input/flant-functions/training_config.json') as f:
    # with open('training_config.json') as f:
        data=f.read()

    data=json.loads(data)

    # file for preprocessing
    input_dir=data['preprocess']['input_dir']
    US_Only=bool(data['preprocess']['US_Only'])

    # file for modeling
    train_path=data['modeling']['train_path']
    test_path=data['modeling']['test_path']
    epochs=int(data['modeling']['epochs'])

    # logging
    logging=bool(data['logging'])
    return input_dir,US_Only,train_path,test_path,epochs,logging

def run_train():
    input_dir,US_Only,train_path,test_path,epochs,logging=load_train_config()
    pre_process(input_dir,US_Only,logging)
    modeling(train_path,test_path,epochs,logging)

run_train()