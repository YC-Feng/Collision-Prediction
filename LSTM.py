'''
LSTM model

For predicting future location of cars

'''


import pandas as pd
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
import math


# - - - - - - - - - - - - - - -Functions- - - - - - - - - - - - - - - - - - - -


def create_dataset(data, n_predictions, n_next):
    '''
    Crteate the datasets include training dataset and label dataset for training
    
    paras:
        @data: Numpy array, original data
        @n_predictions: Int, How much data to use to predict at once
        @n_next: Int, The location of the data going to predict
            e.g. the time interval is 0.1s, 
                if want to predict the location with time interval 1s,
                n_next need to be set 10
    
    return:
        train_X: An array with 2 dimensions
            e.g. [[1,2,3..], [1,2,3...]]
        train_Y: An array with 2 dimensions
            e.g. [[1],[2],[3]]
    '''
    train_X, train_Y = [], []
    
    for i in range(data.shape[0]- n_predictions-n_next) :
        tempx = data[i: (i+n_predictions)]
        train_X.append (tempx)
        tempy = data[i+n_predictions+n_next]
        y=[]
        y.append(tempy)
        train_Y.append (y)
        
    train_X = np.array(train_X, dtype='float64')
    train_Y = np.array(train_Y, dtype='float64')
    
    return train_X, train_Y


def Normalize(data):
    '''
    Normalize the data into 0 and 1
    
    paras:
        @data: original data
    
    return:
        data: normalize data, type as the original data
        normalize: An array with 1 dimensions
            e.g. [high, low] 
    '''

    normalize = np.arange (2, dtype='float64')
    
    low, high = np.percentile(data, [0, 100])
    normalize[0] = low
    normalize[1] = high
    
    delta = high - low
    
    if delta != 0:
        data = (data - low)/delta
        
        
    return data, normalize


def trainModel(train_X, train_Y) :
    '''
    Build LSTM model with one LSTM layer
    
    paras:
        @train_X: training data
        @train_X: label data

    return:
        model: a keras LSTM model
        his: training history
    '''

    model = Sequential()
    model. add(LSTM(
        120,
        activation='tanh',
        input_shape=(100, 1),
        return_sequences=False))
    model.add (Dropout(0.2) )
    
    model.add (Dense(
        train_Y.shape[1]))
    model.add (Activation("tanh"))
    
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    his = model.fit(train_X, train_Y, epochs=80, batch_size=64, verbose=1)
    model.summary()
    
    
    return model, his


def UnNormalize(data, normalize):
    '''
    UnNormalize the predicted data to evaluate the result
    
    paras:
        @data: normalize data
        @normalize: normalize array with 1 dimensions
            e.g. [high, low]

    return:
        data: UnNormalize data, type as the input data
    
    '''
    data = np.array(data, dtype='float64')
    
    low = normalize[0]
    high = normalize[1]
    delta = high - low
    if delta != 0:
        for i in range(0, data.shape[0]) :
            data[i] = data[i]*delta + low
    return data

def Predict(data, timeinterval):
    '''
    Predict the future location of moving objects 
    by using time serie location data
    
    paras:
        @data: 1 dimensions numpy array
        @timeinterval: Int, The prediction time interval (second)

    return:
        result: The MAE measurment result

    '''

    #split data (80% for training)
    num_all = data.shape[0]
    num_for_train = int(np.round(num_all*(8/10), 0))
    data_train = data[:num_for_train]
    data_test = data[num_for_train:]
        
    #take 100 records to predict
    train_num = 100
    
    #time interval (each raw data present 0.1s, so need x10)
    per_num = 10*timeinterval
    
    data, nor = Normalize(data_train)
    train_X, train_Y = create_dataset(data, train_num, per_num)
    
    print('train_X:', train_X.shape)
    print('train_Y:', train_Y.shape)
    
    #training
    model, his = trainModel(train_X, train_Y, )
    loss, acc = model.evaluate(train_X, train_Y, verbose=2)
    print(f'Loss:{loss}, Accuracy:{acc*100}')
    
    #show the training history
    plt.title('train_loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.plot(his.history['loss'])
    
    #prepare testing data 
    data, nor = Normalize(data_test)
    test_X, test_Y = create_dataset(data, train_num, per_num)
    
    
    #predict testing data
    predicted_y = model.predict(test_X)
    
    #UnNormalize testing data
    predicted_y = UnNormalize(predicted_y, nor)
    test_Y = UnNormalize(test_Y, nor)
    
    #transfer data into dataframe to show results
    pred = pd.DataFrame(predicted_y).rename(columns={0:'predict_x'})
    test = pd.DataFrame(test_Y).rename(columns={0:'test_x'})
    
    final = pd.concat([pred, test], axis=1)
    
    #show the predicted error(distance)
    final['distance'] =abs(final['predict_x']-final['test_x'])
    
    result = final['distance'].mean()
    
    #show the MAE
    print('MAE: ', final['distance'].mean())
    
    return result



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


if __name__ == "__main__":
    pass
    
    #suppress scientific notation
    pd.set_option('display.float_format', lambda x : '%.10f' % x)
    
    #read data
    data = pd.read_excel('DataSet.xlsx', header=2)
    
    #the time interval goiuing to predict
    timeinterval = 1
    
    #select the columns we need, in here is only x-axis
    data_x = data.iloc[:, 12].values
    result_x = Predict(data_x, timeinterval)
    
    #select the columns we need, in here is only y-axis
    data_y = data.iloc[:, 13].values
    result_y = Predict(data_y, timeinterval)
    
    #calculate the 2D MAE
    result = math.sqrt(result_x**2 + result_y**2)
    
    #final 2D error result
    print('Prediction result for ', timeinterval, 'second:', result)