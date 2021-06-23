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
### Data Collection
All the data are directly collected from AWS CLI using the following command. ***pulling_data.sh*** is a wrapper for the command to pull all the spot price data in us-east and us-west. Per AWS restriction, we can pull up to 2 months data.
```
aws ec2 describe-spot-price-history --output table\
                                    --product-description "Linux/UNIX (Amazon VPC)"\
                                    --region us-west-1 > ../data/us_west_1.txt
```

### Preprocessing
In this section, we processing the txt file into *DataFrame* and since the price add-on between different OS remain the same in each region, here we ***ONLY*** kept the price history of ***Linux/UNIX***. After that, we split the first 53 days' data as training set, and last 7 days' data as test set. In this section, we also output the price add=on between different OS in different regions so we can make recommendation for users who are interested in OS other than Linux/UNIX.

### Modeling and Model Comparison

### Optimization
In this section, we take user input as constraints, hourly cost as objective function, formulated a optimization problem, we use *Pulp* to solve it. User can feed in different kinds of constraints from the website and can use JSON file to pass the constraints to the function. 

*platform_filter*: User can choose only one or two platform from AWS, Google Cloud and Azure. 

And *opt* is for optimization type, some instances are optimized for computing, others are for storage, so if user have such concerns, we will recommend the optimized instances combination based on that. 

As for *constraints*, some user may not want to use particular instance from particular regions, or these instances are only available for a limit number at that time, user can specify these if they are not satisfied with the first result.

*Min_CPU_num, Min_GPU_num, Min_RAM_Size, Start_Date, End_Date, Region* are self-explanatory.

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

### How to use?
- Install AWS CLI with the help of [this document](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html).
- Set up *aws configure* with the help of [this document](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html).
- 

