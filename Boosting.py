
import pandas as pd
import lightgbm as lgb
from xgboost import XGBRegressor
from matplotlib import pyplot as plt
import numpy as np
import time
import math
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


# - - - - - - - - - - - - - - -Functions- - - - - - - - - - - - - - - - - - - -

def create_dataset(data, axis):
    '''
    Crteate the datasets include training dataset and label dataset for training
    
    paras:
        @data: DataFrame, original data
        @axis: String, define processing x-axis or y-axis of the moving path
        
    
    return:
        train_X: DataFrame, training data
        train_y: DataFrame, trainnig label
        test_X: Series, testing data
        test_y: Series, testing label
    '''
    
    #x-axis or y-axis needs different columns
    if axis == 'X':
        #data = data.iloc[1:3844:,:7:2] #for 1 second prediction
        #data = data.iloc[1:3834, [0, 2, 8, 10]] #for 2 second prediction
        data = data.iloc[1:3824, [0, 2, 12, 14]]
    else:
        #data = data.iloc[1:3844:,1:8:2] #for 1 second prediction
        #data = data.iloc[1:3834, [1, 3, 9, 11]] #for 2 second prediction
        data = data.iloc[1:3824, [1, 3, 13, 15]]
    
    #rename columns name
    data.columns = ['x1', 'x2', 'x3', 'y']
    
    #split training data and testing data
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    
    #split data (~80% for training)
    num_for_train = 3200
    
    train_X = X[:num_for_train]
    train_y = y[:num_for_train]
    test_X = X[num_for_train:]
    test_y = y[num_for_train:]    

    return train_X, train_y, test_X, test_y


def Predict(train_X, train_y, test_X, test_y, m):
    '''
    Predict the future location of moving objects 
    
    paras:
        @train_X: DataFrame, training data
        @train_y: DataFrame, trainnig label
        @test_X: Series, testing data
        @test_y: Series, testing label
        @m: String, X:XGBoost L:lightGBM

    return:
        final: DataFrame, include predicted result, actual result and difference

    '''  

    if m == 'X':
        #build XGBoost model
        model = XGBRegressor()
        #training
        model.fit(train_X,train_y)
        
    elif m == 'L':
        #build lightGBM model
        model = lgb.LGBMRegressor()        
        #training
        model.fit(train_X,train_y)    

    #calculate prediction time
    time_start = time.time() 
    predicted=model.predict(test_X)
    time_end = time.time()    
    time_c= time_end - time_start   
    print('time cost', time_c, 's')
    
    
    #concat final table, predicted result, actual result and difference
    pred = pd.DataFrame(predicted).rename(columns={0:'predict_x'})
    test = pd.DataFrame(test_y).rename(columns={'y':'test_x'})
    test = test.reset_index(drop=True)
    final = pd.concat([pred, test], axis=1)

    #difference between predicted and actual
    final['distance'] =abs(final['predict_x']-final['test_x'])      

    
    return final


def showplot(final):
    '''
    Show the plot of the actual path and predicted path

    paras:
        @final: DataFrame, actual path and predicted path,
                include x-axis and y-axis
    
    return:
        None

    '''
    
    #prepare location lists
    p_x = [x for x in final['predict_x']]
    p_y = [x for x in final['predict_y']]

    a_x = [x for x in final['actual_x']]
    a_y = [x for x in final['actual_y']]
    
    #plot set
    fig, ax = plt.subplots()
    p, = ax.plot(p_x, p_y , 'red', label='predicted')
    t, = ax.plot(a_x, a_y, 'green', label ='actual')
    ax.legend(handles=[p, t])
    
    #inset1 set
    axins = ax.inset_axes((0.2, 0.2, 0.3, 0.2))
    axins.plot(p_x, p_y, color='red')    
    axins.plot(a_x, a_y, color='green')
    
    # set the range
    zone_left = 100
    zone_right = 200
    
    #rescale ratio
    x_ratio = 0.5
    y_ratio = 0.5
    
    #X axis display range
    x = np.hstack((p_x[zone_left:zone_right], a_x[zone_left:zone_right]))
    xlim0 = np.min(x)-(np.max(x)-np.min(x))*x_ratio
    xlim1 = np.max(x)+(np.max(x)-np.min(x))*x_ratio
    
    #y axis display range
    y = np.hstack((p_y[zone_left:zone_right], a_y[zone_left:zone_right]))
    ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
    ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio
    
    #set sub plot
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)
    
    #inset2 set
    axins2 = ax.inset_axes((0.65, 0.5, 0.3, 0.2))    
    axins2.plot(p_x, p_y, color='red')    
    axins2.plot(a_x, a_y, color='green')
    
    #set the range
    zone_left = 400
    zone_right = 500
    
    #rescale ratio
    x_ratio = 0.5
    y_ratio = 0.5
    
    #X axis display range
    x = np.hstack((p_x[zone_left:zone_right], a_x[zone_left:zone_right]))
    xlim0 = np.min(x)-(np.max(x)-np.min(x))*x_ratio
    xlim1 = np.max(x)+(np.max(x)-np.min(x))*x_ratio
    
    #y axis display range
    y = np.hstack((p_y[zone_left:zone_right], a_y[zone_left:zone_right]))
    ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
    ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio
    
    #set sub plot
    axins2.set_xlim(xlim0, xlim1)
    axins2.set_ylim(ylim0, ylim1)
    
    #mark inset
    mark_inset(ax, axins, loc1=3, loc2=1)    
    mark_inset(ax, axins2, loc1=3, loc2=1)
    
    plt.show()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    pass
    
    #suppress scientific notation
    pd.set_option('display.float_format', lambda x : '%.10f' % x)
    
    #read data
    data = pd.read_excel('DataSet_xg.xlsx', header=2)
    
    #the time interval goiuing to predict
    timeinterval = 3
    
    #select the columns we need, in here is only x-axis
    train_X, train_y, test_X, test_y = create_dataset(data, 'X')
    final_table_x = Predict(train_X, train_y, test_X, test_y, 'X')
    result_x = final_table_x['distance'].mean()
    
    
    #select the columns we need, in here is only y-axis
    train_X, train_y, test_X, test_y = create_dataset(data, 'Y')
    final_table_y = Predict(train_X, train_y, test_X, test_y, 'X')
    result_y = final_table_y['distance'].mean()

    
    #calculate the 2D MAE
    result = math.sqrt(result_x**2 + result_y**2)
    
    #final 2D error result
    print('Prediction result for ', timeinterval, 'second:', result)
    
    final = pd.DataFrame()
    final['predict_x'] = final_table_x['predict_x']
    final['predict_y'] = final_table_y['predict_x']
    final['actual_x'] = final_table_x['test_x']
    final['actual_y'] = final_table_y['test_x']
    
    #Show the path plot 
    showplot(final)
