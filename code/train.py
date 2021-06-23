import json
from preprocess import pre_process
from model import modeling

def load_train_config():
    with open('training_config.json') as f:
        train_js=f.read()

    train_js=json.loads(train_js)
    return train_js

def run_train():
    train_js=load_train_config()
    pre_process(train_js['input_dir'],bool(train_js['logging']))
    modeling(train_js['train_path'],train_js['test_path'],train_js['epochs'],train_js['logging'])

if __name__ == '__main__':
    run_train() 

