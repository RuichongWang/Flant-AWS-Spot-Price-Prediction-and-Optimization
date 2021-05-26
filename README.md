# Flant

```
training_config.json     ==   file path or hyperparameters for training
optimization_config.json ==   file path or variables needed for optimization, specifically, user should provide variables in user_input
```
```
Preprocess.py            ==   preprocessing file, take raw data as input, output well split train, test set
Model.py                 ==   main function, take the output of Preprocess.py and output a csv file contains the prediction
Get_Pre.py               ==   wrapper of Preprocess.py and Model.py not necessary
Get_Opt.py               ==   optimization function, take the prediction result and user input (requirement) and output the combination JSON
```
