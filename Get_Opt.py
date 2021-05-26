import json
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
from pulp import *

warnings.filterwarnings("ignore")


def load_config():
    with open('../input/flant-functions/optimization_config.json') as f:
    # with open('optimization_config.json') as f:
        data=f.read()

    data=json.loads(data)

    # input for optimization
    platform_filter=eval(data['optimization']['user_input']['platform_filter'])
    opt=eval(data['optimization']['user_input']['opt'])

    Min_CPU_num=float(data['optimization']['user_input']['Min_CPU_num'])
    Min_GPU_num=float(data['optimization']['user_input']['Min_GPU_num'])
    Min_RAM_Size=float(data['optimization']['user_input']['Min_RAM_Size'])
    Start_Date=data['optimization']['user_input']['Start_Date']
    End_Date=data['optimization']['user_input']['End_Date']

    Region=data['optimization']['user_input']['Region']
    constraints_raw=data['optimization']['user_input']['constraints']
    constraints={}
    for key in constraints_raw:
        k=constraints_raw[key]['region'],constraints_raw[key]['instance']
        constraints[k]=int(constraints_raw[key]['max_num'])

    # global vars for optimization
    TEST=bool(data['optimization']['global_vars']['TEST'])
    plot=bool(data['optimization']['global_vars']['plot'])
    return_json=bool(data['optimization']['global_vars']['return_json'])

    # file for optimization
    aws_price_path=data['optimization']['global_vars']['aws_price_path']
    gcp_price_path=data['optimization']['global_vars']['gcp_price_path']
    azu_price_path=data['optimization']['global_vars']['azu_price_path']

    region_label_path=data['optimization']['global_vars']['region_label_path']
    ins_label_path=data['optimization']['global_vars']['ins_label_path']
    test_set_path=data['optimization']['global_vars']['test_set_path']
    add_on_path=data['optimization']['global_vars']['add_on_path']

    tier_path=data['optimization']['global_vars']['tier_path']
    opt_df_path=data['optimization']['global_vars']['opt_df_path']

    return platform_filter,opt,Min_CPU_num,Min_GPU_num,Min_RAM_Size,Start_Date,End_Date,Region,constraints,TEST,plot,return_json,aws_price_path,gcp_price_path,azu_price_path,region_label_path,ins_label_path,test_set_path,add_on_path,tier_path,opt_df_path

def read_aws_data(aws_price_path,region_label_path,ins_label_path,tier_path,test_set_path,opt_df_path,add_on_path,TEST=False):
    # read tabels
    aws=pd.read_csv(aws_price_path)
    region_label=pd.read_csv(region_label_path)
    ins_label=pd.read_csv(ins_label_path)
    tier=pd.read_csv(tier_path)
    time_df=pd.read_csv(test_set_path,usecols=['Timestamp','Timedel']).drop_duplicates()
    opt_df=pd.read_csv(opt_df_path)
    add_on=pd.read_csv(add_on_path)

    # merge tables
    aws.columns=[' AvailabilityZone', ' InstanceType', 'Timedel', 'Real_AWS_SpotPrice','Price_Prediction']
    aws=ins_label.merge(aws,left_on='InstanceType_label',right_on=' InstanceType').drop(columns=['InstanceType_label',' InstanceType'])
    aws=region_label.merge(aws,left_on='AvailabilityZone_label',right_on=' AvailabilityZone').drop(columns=['AvailabilityZone_label',' AvailabilityZone'])
    aws=tier.merge(aws,on='InstanceType')
    aws=time_df.merge(aws,on='Timedel').drop(columns='Timedel')
    aws=opt_df.merge(aws,on='instanceOptimized_label').drop(columns='instanceOptimized_label')
    aws=aws.merge(add_on,on=['InstanceType','AvailabilityZone'])

    # clean tables
    aws['gpu']=np.where(aws['gpu']==-1,0,aws['gpu'])
    aws['Timestamp']=pd.to_datetime(aws['Timestamp'].transform(lambda x:x.split('+')[0]))
    if TEST:
        np.random.seed(42)
        weights=np.random.random(len(aws))/500
        bi=[int(x<0.93) for x in np.random.random(len(aws))]
        aws['Price_Prediction']=aws['Real_AWS_SpotPrice']*(1+weights*bi)
        
    aws['Price_Prediction']+=aws['SUSE_Linux_price_diff']   # price add on
    aws=aws[['Timestamp', 'InstanceType','AvailabilityZone', 'cpu', 'gpu', 'RAM','instanceOptimized', 'Price_Prediction']]


    return aws

def read_gcp_data(gcp_price_path,add_on_path):
    gcp_raw=pd.read_csv(gcp_price_path)
    gcp_raw=gcp_raw[gcp_raw.AvailabilityZone.str.startswith('us-')]

    gcp_raw['gpu']=0
    gcp_raw['Price_Prediction']+=pd.read_csv(add_on_path).SUSE_Linux_price_diff.mean()   # price add on
    gcp_raw['instanceOptimized']='General Purpose'
    gcp_raw=gcp_raw[['InstanceType', 'AvailabilityZone', 'cpu', 'gpu', 'RAM','instanceOptimized', 'Price_Prediction']]
    return gcp_raw

def read_azu_data(azu_price_path,add_on_path):
    azure_raw=pd.read_csv(azu_price_path)
    azure_raw=azure_raw[azure_raw.os=='SUSE'].drop(columns=['os','storage'])
    azure_raw.columns=['AvailabilityZone','InstanceType', 'cpu', 'RAM', 'Price_Prediction']

    azure_raw['gpu']=0
    azure_raw['Price_Prediction']+=pd.read_csv(add_on_path).SUSE_Linux_price_diff.mean()   # price add on
    azure_raw['instanceOptimized']='General Purpose'
    azure_raw=azure_raw[['InstanceType', 'AvailabilityZone', 'cpu', 'gpu', 'RAM','instanceOptimized', 'Price_Prediction']]

    return azure_raw

def opt_one_hr(aws,gcp,azu,sample_stamp,Min_CPU_num,Min_GPU_num,Min_RAM_Size,constraints=False,platform_filter=None):
    sample_df=aws[aws.Timestamp==sample_stamp]
    if platform_filter=='aws':
        sample_df=aws
    elif platform_filter=='gcp':
        sample_df=gcp
    elif platform_filter=='azu':
        sample_df=azu                
    else:
        sample_df=pd.concat((aws,gcp,azu))

    sample_df['merge']=sample_df.AvailabilityZone+'_'+sample_df.InstanceType
    sample_df['cpu']=sample_df['cpu'].astype(int)
    sample_df['gpu']=sample_df['gpu'].astype(int)
    sample_df['RAM']=sample_df['RAM'].astype(str).str.replace(',','').astype(float)
    sample_df.sort_values('merge',inplace=True)

    probA=LpProblem("Problem A",LpMinimize)
    # Define Parameters and parameter dictionaries 
    region_ins=sorted(sample_df['merge'].unique())
    cpus=dict(sample_df[['merge','cpu']].values)
    gpus=dict(sample_df[['merge','gpu']].values)
    rams=dict(sample_df[['merge','RAM']].values)
    prices=dict(sample_df[['merge','Price_Prediction']].values)

    # Define decision variables 
    decisions=LpVariable.dicts("x",region_ins,lowBound=0,cat='Integer')

    # Define objective function    
    costs = lpSum([decisions[i]*prices[i] for i in region_ins])
    probA+=costs

    # Define constraints       
    probA+=lpSum([decisions[i]*cpus[i] for i in region_ins]) >= Min_CPU_num,"CPU"
    probA+=lpSum([decisions[i]*gpus[i] for i in region_ins]) >= Min_GPU_num,"GPU"
    probA+=lpSum([decisions[i]*rams[i] for i in region_ins]) >= Min_RAM_Size,"RAM"
    
    if constraints:
        for region,ins in constraints:
            merged=region+'_'+ins
            probA+=decisions[merged] <= constraints[(region,ins)]
            
    probA.solve()
    output=[]
    for i in region_ins:
        output.append(decisions[i].varValue)
    return sample_df,output,value(probA.objective)
    
def optimizer(aws,gcp,azu,Min_CPU_num,Min_GPU_num,Min_RAM_Size,Start_Date,End_Date,Region=None,platform_filter=None,opt=None,plot=True,constraints=False,return_json=True):
    aws=aws[(aws.Timestamp>=pd.to_datetime(Start_Date)) & (aws.Timestamp<=pd.to_datetime(End_Date))]
    sample_stamps=aws.Timestamp.unique()

    # optimization type filtering 
    if opt:
        if opt not in ['Acceleration','Compute','Memory','Storage']:
            print("Wrong Optimization Type! Supported Optimization Types Are: 'Acceleration','Compute','Memory','Storage'.")
            return None
                
        aws=aws[aws.instanceOptimized.str.contains(opt)]
        gcp=gcp[gcp.instanceOptimized.str.contains(opt)]
        azu=azu[azu.instanceOptimized.str.contains(opt)]

    # region filtering
    if Region:
        region_dict={
                    'us-east':['us-east-1a', 'us-east-1b', 'us-east-1c', 'us-east-1d','us-east-1e', 
                               'us-east-1f','us-east4', 'us-east1','East US', 'East US 2',],
                    'us-central':['us-central1', 'Central US'],
                    'us-north-central':['North Central US'],
                    'us-south-central':['South Central US'],
                    'us-west-central':['West Central US'],
                    'us-west':['us-west1', 'us-west2','us-west3', 'us-west4','West US', 'West US 2']
                    }

        using=[]

        if type(Region)==list:
            for region in Region:
                using.extend(region_dict[region])
        else: using=region_dict[Region]
        aws=aws[aws.AvailabilityZone.isin(using)]
        gcp=gcp[gcp.AvailabilityZone.isin(using)]
        azu=azu[azu.AvailabilityZone.isin(using)]

    # optimization
    simple_res=[]
    for sample_stamp in sample_stamps:
        sample_df,output,cost=opt_one_hr(aws,gcp,azu,sample_stamp,Min_CPU_num,Min_GPU_num,Min_RAM_Size,constraints,platform_filter)
        ins=sorted(sample_df['merge'].unique())
        
        output_df=pd.DataFrame(output,columns=['Number of Purchase'],index=ins)
        output_df=output_df[output_df['Number of Purchase']>0]
        
        nums=output_df.T.values[0]
        ins=output_df.index
        simple_res.append([sample_stamp,cost]+[str(nums[i])+'_'+str(ins[i]) for i in range(len(nums))])
        
    simple_res=pd.DataFrame(simple_res)
    simple_res.columns=['TimeStamp','Cost']+['InstanceType_%s'%x for x in range(1,simple_res.shape[1]-1)]
    inss=['InstanceType_%s'%x for x in range(1,simple_res.shape[1]-1)]
    simple_res['merged']=simple_res[inss].apply(lambda x:'+'.join(x.astype(str)),axis=1)

    # plotting
    if plot:
        gap=simple_res.Cost.max()-simple_res.Cost.min()
        
        fig = px.bar(simple_res, x='TimeStamp', y='Cost',color="merged",text='Cost')
        fig.update_traces(textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig.update_yaxes(ticklabelposition="inside top", title='Best Cost')
        fig.update_yaxes(range=[simple_res.Cost.min()-gap*0.3,simple_res.Cost.max()+gap*0.3], row=1, col=1)
        fig.update_layout(legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    ))
        fig.show()  
            
        simple_res['lag']=simple_res['merged'].shift(1)
        simple_res['same']=simple_res['lag']==simple_res['merged']
        simple_res=simple_res[~simple_res['same']].drop(['merged','lag','same'],axis=1)

    # return JSON
    if return_json:
        simple_res_arr=simple_res.values
        temp_json={}
        for i in range(len(simple_res_arr)):
            timestamp=str(simple_res_arr[i][0])
            temp_json[timestamp]={}
            temp_json[timestamp]['CombinationCost']=simple_res_arr[i][1]
            temp_json[timestamp]['InstanceComb']={}

            for i,num_region_ins in enumerate(simple_res_arr[i][2:]):
                if type(num_region_ins)==str: 
                    num=int(float(num_region_ins.split('_')[0]))
                    region=num_region_ins.split('_')[1]
                    ins=num_region_ins.split('_')[2]

                    temp_json[timestamp]['InstanceComb']['InstanceType_%s'%(i+1)]={'AvailabilityZone':region,
                                                                                   'InstanceType':ins,
                                                                                   'Number':num}      
        return temp_json
    else:
        return simple_res

def run_opt():
    platform_filter,opt,Min_CPU_num,Min_GPU_num,Min_RAM_Size,Start_Date,End_Date,Region,constraints,TEST,plot,return_json,aws_price_path,gcp_price_path,azu_price_path,region_label_path,ins_label_path,test_set_path,add_on_path,tier_path,opt_df_path=load_config()
    aws=read_aws_data(aws_price_path,region_label_path,ins_label_path,tier_path,test_set_path,opt_df_path,add_on_path,TEST=TEST)
    gcp=read_gcp_data(gcp_price_path,add_on_path)
    azu=read_azu_data(azu_price_path,add_on_path)
    res=optimizer(aws,gcp,azu,Min_CPU_num,Min_GPU_num,Min_RAM_Size,Start_Date,End_Date,Region=Region,platform_filter=platform_filter,opt=opt,plot=plot,constraints=constraints,return_json=return_json)
    return res

run_opt()