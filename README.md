# Flant
## Repository contents
```
training_config.json     ==   file path or hyperparameters for training
optimization_config.json ==   file path or variables needed for optimization, specifically, user should provide variables in user_input
user_config.json         ==   user requirements, usually are constraints for region, CPU number or RAM size.
```
```
loop.sh                  ==   bash code for making all the code (pulling data and training models) run 24/7
pulling_data.sh          ==   bash code for pulling spot price history directly from AWS official website
Preprocess.py            ==   preprocessing file, take raw data as input, output well split train, test set
Model.py                 ==   main function, take the output of Preprocess.py and output a csv file contains the prediction
Get_Pre.py               ==   wrapper of Preprocess.py and Model.py not necessary
Get_Opt.py               ==   optimization function, take the prediction result and user input (requirement) and output the combination JSON
```

## Repository summary
### How to use?
- Install AWS CLI with the help of [this document](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html).
- Set up *aws configure* with the help of [this document](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html).
- (Ignore this step if you are using it locally) Upload all the code to your server.
- Pulling the data using `sh pulling_data.sh`, this may take serveral minutes.
- Preprocess the data using `python3 Preprocess.py`.
- Train models and get predictions by using `python3 Get_Pre.py`.
- Get optimization result by using `python3 Get_Opt.py`.

### Sample Output
```
{'2021-06-10 06:00:00': {
    'CombinationCost': 1.5751331341739343, 
    'InstanceComb': {
        'InstanceType_1': {
            'AvailabilityZone': 'us-central1', 
            'InstanceType': 'm1-ultramem-40', 
            'Number': 1}, 
        'InstanceType_2': {
            'AvailabilityZone': 'us-west-1a', 
            'InstanceType': 't3a.micro', 
            'Number': 31}, 
        'InstanceType_3': {
            'AvailabilityZone': 'us-west-1a', 
            'InstanceType': 't4g.micro', 
            'Number': 5}, 
        'InstanceType_4': {
            'AvailabilityZone': 'us-west-1c', 
            'InstanceType': 't4g.micro', 
            'Number': 3}
        }
    }
}
```

### Data Collection
All the data are directly collected from AWS CLI using the following command. ***pulling_data.sh*** is a wrapper for the command to pull all the spot price data in us-east and us-west. Per AWS restriction, we can pull up to 2 months data.
```
aws ec2 describe-spot-price-history --output table\
                                    --product-description "Linux/UNIX (Amazon VPC)"\
                                    --region us-west-1 > ../data/us_west_1.txt
```

### Preprocessing
In this section, we processing the txt file into *DataFrame* and since the price add-on between different OS remain the same in each region, here we ***ONLY*** kept the price history of ***Linux/UNIX***. After that, we split the first 53 days' data as training set, and last 7 days' data as test set. In this section, we also output the price add=on between different OS in different regions so we can make recommendation for users who are interested in OS other than Linux/UNIX.

### Machine Learning Modelling
Various Machine Learning and Deep Learning models including SARIMAX, Prophet, MLP, etc. were employed to predict the price of spot/preemptible instances. Finally, the one with the best scores was chosen for deployment.
The following table shows the evaluation scores of the significantly contributing models that we examined.
| Features        | Target         | Model    | MAE    | RMSE   | MAPE    |
|-----------------|----------------|----------|--------|--------|---------|
| Feature Set   1 | SpotPrice(t+1) | [CNN-LSTM](https://machinelearningmastery.com/cnn-long-short-term-memory-networks/) | 0.0004 | 0.0027 | [0.4834](https://github.com/RuichongWang/Flant/blob/main/Notebooks/cnn-lstm-the-chosen-one.ipynb)  |
| Feature Set   2 | SpotPrice      | [LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | 0.0051 | 0.1140  | [2.6201](https://github.com/RuichongWang/Flant/blob/main/Notebooks/LSTM.ipynb)|
| Feature Set   3 | SpotPrice      | [LSTM-2](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | 0.0051 | 0.1140  | [3.4693](https://github.com/RuichongWang/Flant/blob/main/Notebooks/LSTM_Feature%20Set%203.ipynb)|
| Feature Set   3 | SpotPrice      | [LGBM](https://github.com/microsoft/LightGBM)     | 0.0031 | 0.0079 | [1.7423](https://github.com/RuichongWang/Flant/blob/main/Notebooks/LGBM.ipynb)  |
| Feature Set   1 | SpotPrice(t+1) | [CNN](https://towardsdatascience.com/basics-of-the-classic-cnn-a3dce1225add)    | 0.0016 | 0.0038  | [1.2246](https://github.com/RuichongWang/Flant/blob/main/Notebooks/convolutional-neural-network.ipynb)  |
| Feature Set   4 | SpotPrice      | [PyCaret](https://www.pycaret.org/tutorials/html/REG101.html) | 0.0169 | 0.2382 | [1.3099](https://github.com/RuichongWang/Flant/blob/main/Notebooks/pycaret-randomforest-with-cv-3.ipynb) |

Please see the details on different kinds of time-series and statistical features we used to evaluate the models.

| Feature Identity | Features Used    |
|------------------|------------------|
| Feature Set 1    | `SpotPrice(t-3)`, `SpotPrice(t-2)`, `SpotPrice(t-1)`, `SpotPrice(t)`  |
| Feature Set 2    | `InstanceType`, `Availability`, `Date`, `Day`, `Hour`, `Timedel`, `Weekend_YorN`, `DayofYear`, `4MA`, `4SD`, `24MA`, `24SD`, `upperband`, `lowerband`, `RC_24` |
| Feature Set 3    | `AvailabilityZone`, `InstanceType`, `Timedel`, `Day`, `Hour`, `Weekend_YorN`, `DayofYear` |
| Feature Set 4    | `SpotPrice(t-2)`, `SpotPrice(t-1)`, `SpotPrice(t)` |


**`4MA`**: 4 hour's rolling mean price

**`4SD`**: 4 hour's rolling standard deviation of price

**`24MA`**: 24 hour's rolling mean price

**`24SD`**: 24 hour's rolling standard deviation of price

**`upperband`**: 24 hour's rolling mean price + 24 hour's rolling standard deviation of price

**`lowerband`**: 24 hour's rolling mean price - 24 hour's rolling standard deviation of price

**`RC_24`**: Percentage of change in price in 24 hours


*CNN-LSTM performed the best on the AWS spot price dataset and we concluded the time-series price forecast by implementing this one.*

### Optimization
In this section, we take user input as constraints, hourly cost as objective function, formulated a optimization problem, we use *Pulp* to solve it. User can feed in different kinds of constraints from the website and can use JSON file to pass the constraints to the function. 

*platform_filter*: User can choose only one or two platform from AWS, Google Cloud and Azure. 

And *opt* is for optimization type, some instances are optimized for computing, others are for storage, so if user have such concerns, we will recommend the optimized instances combination based on that. 

As for *constraints*, some user may not want to use particular instance from particular regions, or these instances are only available for a limit number at that time, user can specify these if they are not satisfied with the first result.

*`Min_CPU_num`, `Min_GPU_num`, `Min_RAM_Size`, `Start_Date`, `End_Date`, `Region`* are self-explanatory.

```
{
    "platform_filter":"",
    "opt":"",
    "Min_CPU_num":0,
    "Min_GPU_num":0,
    "Min_RAM_Size":1000,
    "Start_Date":"2021-6-10 6:00:00",
    "End_Date":"2021-6-10 18",
    "Region":["us-west","us-central"],
    "plot":"True",
    "constraints":{
        "constraint_1":{
            "region":"us-west-1a",
            "instance":"t4g.micro",
            "max_num":5
        },
        "constraint_2":{
            "region":"us-west-1c",
            "instance":"t4g.micro",
            "max_num":3
        }
    }
}
```
