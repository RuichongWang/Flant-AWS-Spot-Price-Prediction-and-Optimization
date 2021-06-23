import os
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint)
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed,Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential

warnings.filterwarnings('ignore')


def modeling(train_path,test_path,epochs,logging=False):
    # config
    window = 3
    lag = 1
    subsequences = 1
    lag_size = 1 

    # seed everything
    SEED = 0
    tf.random.set_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    np.random.seed(0)
    tf.random.set_seed(0)

    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1, 
        inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(), 
        config=session_conf
    )
    K.set_session(sess)

    def feat_eng(df):
        df.drop('Timestamp', axis = 1, inplace = True)
        df = df.reset_index(drop = True)
        return df

    def series_to_supervised(data, window=1, lag=1, dropnan=True):
        cols, names = list(), list()
        for i in range(window, 0, -1):
            cols.append(data.shift(i))
            names += [('%s(t-%d)' % (col, i)) for col in data.columns]
        cols.append(data)
        names += [('%s(t)' % (col)) for col in data.columns]
        cols.append(data.shift(-lag))
        names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def reduce_mem_usage(df):
        for col in df.columns:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        return df

    def mem_opt(t_set):
        
        cols = t_set.columns.tolist()
        t_set = t_set[cols].astype(str)
        t_set = t_set.astype(float)
        t_set = t_set.to_numpy()
        return t_set

    # load data
    dataset_train = pd.read_csv(train_path)
    train = dataset_train.copy()

    dataset_train['Timestamp'] = pd.to_datetime(dataset_train['Timestamp'], format = '%Y-%m-%d %H:%M:%S.%f')
    dataset_train['Date']= pd.to_datetime(dataset_train['Timestamp']).apply(lambda x: x.date())
    dataset_train["Date"] = pd.to_datetime(dataset_train["Date"], format = '%Y-%m-%d')

    dataset_test = pd.read_csv(test_path)
    dataset_test['Timestamp'] = pd.to_datetime(dataset_test['Timestamp'], format = '%Y-%m-%d %H:%M:%S.%f')
    dataset_test['Date']= pd.to_datetime(dataset_test['Timestamp']).apply(lambda x: x.date())
    dataset_test["Date"] = pd.to_datetime(dataset_test["Date"], format = '%Y-%m-%d')

    dataset_train = feat_eng(dataset_train)
    dataset_test = feat_eng(dataset_test)
    dataset_ = pd.concat([dataset_train, dataset_test])

    # process data
    series = series_to_supervised(dataset_.drop('Date', axis=1), window=window, lag=lag)
    columns_to_drop = [('%s(t+%d)' % (col, lag)) for col in ['AvailabilityZone', 'InstanceType', 'Timedel']]
    for i in range(window, 0, -1):
        columns_to_drop += [('%s(t-%d)' % (col, i)) for col in ['AvailabilityZone', 'InstanceType', 'Timedel']]
    series.drop(columns_to_drop, axis=1, inplace=True)
    series.drop(['AvailabilityZone(t)', 'InstanceType(t)', 'Timedel(t)'], axis=1, inplace=True)

    labels_col = 'SpotPrice(t+%d)' % lag_size
    labels = series[[labels_col]]
    series = series.drop(labels_col, axis=1)

    X_train = series[:len(dataset_train)- window]
    Y_train = labels[:len(dataset_train)- window]
    X_valid = series[len(dataset_train)- window:]
    Y_valid = labels[len(dataset_train)- window:]

    X_train=reduce_mem_usage(X_train)
    Y_train=reduce_mem_usage(Y_train)
    X_valid=reduce_mem_usage(X_valid)
    Y_valid=reduce_mem_usage(Y_valid)

    X_train = mem_opt(X_train)
    X_valid = mem_opt(X_valid)

    X_train_series = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_valid_series = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))
    print('Train set shape', X_train_series.shape)
    print('Validation set shape', X_valid_series.shape)

    timesteps = X_train_series.shape[1]//subsequences
    X_train_series_sub = X_train_series.reshape((X_train_series.shape[0], subsequences, timesteps, 1))
    X_valid_series_sub = X_valid_series.reshape((X_valid_series.shape[0], subsequences, timesteps, 1))
    print('Train set shape', X_train_series_sub.shape)
    print('Validation set shape', X_valid_series_sub.shape)

    # model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)

    model_cnn_lstm = Sequential()
    model_cnn_lstm.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), 
                                                input_shape=(None,X_train_series.shape[1],X_train_series.shape[2])))
    model_cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=1)))
    model_cnn_lstm.add(TimeDistributed(Flatten()))
    model_cnn_lstm.add(LSTM(50, activation='relu'))
    model_cnn_lstm.add(Dense(1))
    model_cnn_lstm.compile(loss='mae', optimizer='adam')

    # train
    cnn_lstm_history = model_cnn_lstm.fit(X_train_series_sub, Y_train, validation_data=(X_valid_series_sub, Y_valid), epochs=epochs, verbose=1, callbacks=[es, mcp])

    # predict
    cnn_lstm_valid_pred = model_cnn_lstm.predict(X_valid_series_sub)

    comparison = pd.read_csv(test_path)
    comparison = comparison[:len(dataset_test)-lag]
    comparison['Predicted_Price'] = cnn_lstm_valid_pred
    comparison.columns =['AvailabilityZone', 'InstanceType', 'Timestamp', 'Timedel', 'Actual_Price', 'Predicted_Price']
    comparison.to_csv('../data/Prediction.csv', index=False)
    
    # test errors
    errors = comparison['Actual_Price'] - comparison['Predicted_Price']
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    mape = np.abs(errors/comparison['Actual_Price']).mean() * 100
    r2 = r2_score(comparison['Actual_Price'], comparison['Predicted_Price'])
    print('Mean Absolute Error: {:.4f}'.format(mae))
    print('Root Mean Square Error: {:.4f}'.format(rmse))
    print('Mean Absolute Percentage Error: {:.4f}'.format(mape))
    print('R2 Score: {:.4f}'.format(r2))
