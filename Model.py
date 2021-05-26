import datetime
import holidays
import numpy as np
import pandas as pd
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

# # file paths
# train_path='../input/sample-prediction/Predictions.csv'
# test_path='../input/flant-gcp-data/GCP.csv'
# epochs=1
# verbose=1

def modeling(train_path,test_path,epochs,logging=False):
    dataset_train=pd.read_csv(train_path)
    dataset_train['Timestamp'] = pd.to_datetime(dataset_train['Timestamp'], format = '%Y-%m-%d %H:%M:%S.%f')
    dataset_train['Date']= pd.to_datetime(dataset_train['Timestamp']).apply(lambda x: x.date())
    dataset_train["Date"] = pd.to_datetime(dataset_train["Date"], format = '%Y-%m-%d')

    dataset_test=pd.read_csv(test_path)
    dataset_test['Timestamp'] = pd.to_datetime(dataset_test['Timestamp'], format = '%Y-%m-%d %H:%M:%S.%f')
    dataset_test['Date']= pd.to_datetime(dataset_test['Timestamp']).apply(lambda x: x.date())
    dataset_test["Date"] = pd.to_datetime(dataset_test["Date"], format = '%Y-%m-%d')

    def feat_eng(df):
        df['Date'] = df['Timestamp'].dt.date
        df['Date'] = pd.to_datetime(df['Date'])
        df['Day'] = df['Timestamp'].dt.day
        df['Hour'] = df['Timestamp'].dt.hour

        df['Weekend_YorN'] = ((df['Timestamp'].dt.dayofweek) // 5 == 1).astype(float)
        df['DayofYear'] = df['Timestamp'].dt.dayofyear

        hd = []
        for ptr in holidays.US(years = 2021).items():
            hd.append(ptr[0])

        df.drop('Timestamp', axis = 1, inplace = True)
        df = df.reset_index(drop = True)
        return df

    if logging: print(str(datetime.datetime.now()).split('.')[0],'feature engineering...')
    dataset_train= feat_eng(dataset_train)
    dataset_test= feat_eng(dataset_test)

    dataset_train['Actual'] =dataset_train['SpotPrice']
    dataset_train= dataset_train.drop(['SpotPrice'], axis=1)

    dataset_train['SpotPrice'] =dataset_train['Actual']
    dataset_train= dataset_train.drop(['Actual'], axis=1)

    dataset_test['Actual'] =dataset_test['SpotPrice']
    dataset_test= dataset_test.drop(['SpotPrice'], axis=1)
    dataset_test['SpotPrice'] =dataset_test['Actual']
    dataset_test= dataset_test.drop(['Actual'], axis=1)

    fsub=dataset_test[1:].copy()
    train_hours = len(dataset_train)
    dataset_train = pd.concat([dataset_train, dataset_test])

    dataset_train = dataset_train[['AvailabilityZone', 'InstanceType', 'Timedel', 'Day', 'Hour', 'Weekend_YorN', 'DayofYear', 'SpotPrice']].astype(str)
    dataset_train = dataset_train.astype(float)
    training_set = dataset_train.to_numpy()

    # Feature Scaling
    sc = StandardScaler()
    sc.fit(training_set)
    training_set_scaled = sc.transform(training_set)

    def to_supervised(dataset_train,dropNa = True,lag = 1):
        df = pd.DataFrame(dataset_train)
        column = []
        column.append(df)
        for i in range(1,lag+1):
            column.append(df.shift(-i))
        df = pd.concat(column,axis=1)
        df.dropna(inplace = True)
        features = dataset_train.shape[1]
        df = df.values
        supervised_data = df[:,:features*lag]
        supervised_data = np.column_stack( [supervised_data, df[:,features*lag]])
        return supervised_data

    timeSteps = 2
    if logging: print(str(datetime.datetime.now()).split('.')[0],'to supervised...')
    supervised = to_supervised(training_set_scaled,lag=timeSteps)

    # spiltting the data
    features =dataset_train.shape[1]

    X = supervised[:,:features*timeSteps]
    y = supervised[:,features*timeSteps]

    x_train = X[:train_hours,:]
    x_test = X[train_hours:,:]
    y_train = y[:train_hours]
    y_test = y[train_hours:]

    x_train = x_train.reshape(x_train.shape[0], timeSteps, features)
    x_test = x_test.reshape(x_test.shape[0], timeSteps, features)

    #define the model
    if logging: print(str(datetime.datetime.now()).split('.')[0],'modeling...')
    model = Sequential()
    model.add( LSTM( 50, input_shape = ( timeSteps,x_train.shape[2]) ) )
    model.add( Dense(1) )

    model.compile( loss = "mae", optimizer = "adam")

    es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=0)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0)
    mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True)
    tb = TensorBoard('logs')

    _ = model.fit(x_train, y_train, shuffle=True, epochs=epochs, callbacks=[es, rlr, mcp, tb],validation_data = (x_test,y_test), verbose=logging, batch_size=256)

    #scale back the prediction to orginal scale
    if logging: print(str(datetime.datetime.now()).split('.')[0],'predicting...')
    prediction= model.predict(x_test)
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[2]*x_test.shape[1])

    inv_new = np.concatenate( (prediction, x_test[:,-7:] ) , axis =1)
    inv_new = sc.inverse_transform(inv_new)
    final_prediction = inv_new[:,-1]

    y_test = y_test.reshape( len(y_test), 1)

    inv_new = np.concatenate( (y_test, x_test[:,-7:] ) ,axis = 1)
    inv_new = sc.inverse_transform(inv_new)
    actual_prediction = inv_new[:,-1]

    def evaluate_prediction(final_prediction, actual_prediction, model):
        errors = actual_prediction - final_prediction
        mse = np.square(errors).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(errors).mean()
        mape = np.abs(errors/actual_prediction).mean() * 100

        print('Mean Absolute Error: {:.4f}'.format(mae))
        print('Root Mean Square Error: {:.4f}'.format(rmse))
        print('Mean Absolut Percentage Error: {:.4f}'.format(mape))

    evaluate_prediction(prediction, y_test, 'LSTM')

    col1 = pd.DataFrame(final_prediction, columns=['Prediction'])
    col2 = pd.DataFrame(actual_prediction, columns=['SpotPrice'])

    results = pd.concat([col1, col2], axis=1)
    results.to_csv('results', index=False)

    final_prediction = fsub.copy()
    predictions = pd.DataFrame({'AvailabilityZone':final_prediction.AvailabilityZone,' InstanceType':final_prediction.InstanceType,'Timedel':final_prediction.Timedel,'SpotPrice':final_prediction.SpotPrice,'Price Prediction':results.Prediction})
    predictions.to_csv('Prediction.csv', index=False)
