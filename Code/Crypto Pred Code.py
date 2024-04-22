import os
import pandas as pd
import numpy as np
import math
import datetime as dt

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from itertools import cycle
import plotly.graph_objects as go # add details
import plotly.express as px
from plotly.subplots import make_subplots

!pip install colorama
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore #for adding colours

#Bitcoin is a decentralized digital currency that operates on a peer-to-peer network without a central authority.
bitcoindf = pd.read_csv(r"/content/BTC-USD.csv")
bitcoindf = bitcoindf.rename(columns={'Date': 'date','Open':'open','High':'high','Low':'low','Close':'close',
                                'Adj Close':'adj_close','Volume':'volume'})
bitcoindf.head()
bitcoindf.shape

#Dogecoin
#Dogecoin is primarily used for tipping users on Reddit and Twitter,
#but it is also accepted as a method of payment by a few dozen merchants.
dogecoindf = pd.read_csv(r"/content/DOGE-USD.csv")
dogecoindf = dogecoindf.rename(columns={'Date': 'date','Open':'open','High':'high','Low':'low','Close':'close',
                                'Adj Close':'adj_close','Volume':'volume'})
dogecoindf.head()

#Ethereum
#Ethereum operates on a decentralized computer network, or distributed ledger called a blockchain, which manages and tracks the currency.
#It can be useful to think of a blockchain like a running receipt of every transaction that's ever taken place in the cryptocurrency.

ethereumdf = pd.read_csv(r"/content/ETH-USD.csv")
ethereumdf = ethereumdf.rename(columns={'Date': 'date','Open':'open','High':'high','Low':'low','Close':'close',
                                'Adj Close':'adj_close','Volume':'volume'})
ethereumdf.head()

#Cardano
#The cardano blockchain can be used to build smart contracts, and in turn, create decentralized applications and protocols. Additionally,
#the ability to send and receive funds instantly through, for minimal fees, have many applications in the world of business and finance.

cardanodf = pd.read_csv(r"/content/ADA-USD.csv")
cardanodf = cardanodf.rename(columns={'Date': 'date','Open':'open','High':'high','Low':'low','Close':'close',
                                'Adj Close':'adj_close','Volume':'volume'})
cardanodf.head()

bitcoindf = bitcoindf.fillna(method = 'ffill')
dogecoindf = dogecoindf.fillna(method = 'ffill')
ethereumdf = ethereumdf.fillna(method = 'ffill')
cardanodf = cardanodf.fillna(method = 'ffill')

#Convert Date column into Datatime format
bitcoindf['date'] = pd.to_datetime(bitcoindf.date)
dogecoindf['date'] = pd.to_datetime(dogecoindf.date)
ethereumdf['date'] = pd.to_datetime(ethereumdf.date)
cardanodf['date'] = pd.to_datetime(cardanodf.date)

#Plotting close price of Bitcoin, Cardano, Dogecoin and Ethereum

fig = plt.figure(figsize = (15,10))

plt.subplot(2, 2, 1)
plt.plot(bitcoindf['date'], bitcoindf['close'], color="red")
plt.title('Bitcoin Close Price')

plt.subplot(2, 2, 2)
plt.plot(cardanodf['date'], cardanodf['close'], color="black")
plt.title('Cardano Close Price')

plt.subplot(2, 2, 3)
plt.plot(dogecoindf['date'], dogecoindf['close'], color="orange")
plt.title('Dogecoin Close Price')

plt.subplot(2, 2, 4)
plt.plot(ethereumdf['date'], ethereumdf['close'], color="green")
plt.title('Ethereum Close Price')

#Plotting only 2020-2021 year close price of Bitcoin, Cardano, Dogecoin and Ethereum
last1year_bitcoindf = bitcoindf[bitcoindf['date'] > '09-2020']
last1year_cardanodf = cardanodf[cardanodf['date'] > '09-2020']
last1year_dogecoindf = dogecoindf[dogecoindf['date'] > '09-2020']
last1year_ethereumdf = ethereumdf[ethereumdf['date'] > '09-2020']

fig = plt.figure(figsize = (15,10))
fig.suptitle("Last 1 year close prices of Bitcoin, Cardano, Dogecoin, Ethereum", fontsize=16)


plt.subplot(4, 1, 1)
plt.plot(last1year_bitcoindf['date'], last1year_bitcoindf['close'], color="red")
plt.legend("B")

plt.subplot(4, 1, 2)
plt.plot(last1year_cardanodf['date'], last1year_cardanodf['close'], color="black")
plt.legend("C")

plt.subplot(4, 1, 3)
plt.plot(last1year_dogecoindf['date'], last1year_dogecoindf['close'], color="orange")
plt.legend("D")

plt.subplot(4, 1, 4)
plt.plot(last1year_ethereumdf['date'], last1year_ethereumdf['close'], color="green")
plt.legend("E")

#histogram for all the coloumns of bitcoin dataset.S
bitcoin_numeric = bitcoindf.select_dtypes(exclude = ["bool"])
bitcoin_numeric.hist(figsize=(18,12))
plt.show()

closedf = bitcoindf[['date','close']]
print("Shape of close dataframe:", closedf.shape)

closedf = closedf[closedf['date'] > '2020-09-13']
close_stock = closedf.copy()
print("Total data for prediction: ",closedf.shape[0])

del closedf['date']
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
print(closedf.shape)

training_size=int(len(closedf)*0.70)
test_size=len(closedf)-training_size
train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1] #splitting
print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)

fig, ax = plt.subplots(figsize=(15, 6))
sns.lineplot(x = close_stock['date'][:255], y = close_stock['close'][:255], color = 'black')
sns.lineplot(x = close_stock['date'][255:], y = close_stock['close'][255:], color = 'red')

# Formatting
ax.set_title('Train & Test data', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
ax.set_xlabel('Date', fontsize = 16, fontdict=dict(weight='bold'))
ax.set_ylabel('Price', fontsize = 16, fontdict=dict(weight='bold'))
plt.tick_params(axis='y', which='major', labelsize=16)
plt.tick_params(axis='x', which='major', labelsize=16)
plt.legend(loc='upper right' ,labels = ('train', 'test'))

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)

from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=1000)
my_model.fit(X_train, y_train, verbose=False)

predictions = my_model.predict(X_test)
print("Mean Absolute Error - MAE : " + str(mean_absolute_error(y_test, predictions)))
print("Root Mean squared Error - RMSE : " + str(math.sqrt(mean_squared_error(y_test, predictions))))

train_predict=my_model.predict(X_train)
test_predict=my_model.predict(X_test)

train_predict = train_predict.reshape(-1,1)
test_predict = test_predict.reshape(-1,1)

print("Train data prediction:", train_predict.shape)
print("Test data prediction:", test_predict.shape)

# Transform back to original form

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))

# shift train predictions for plotting

look_back=time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
print("Train predicted data: ", trainPredictPlot.shape)

# shift test predictions for plotting
testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
print("Test predicted data: ", testPredictPlot.shape)

names = cycle(['Original close price','Train predicted close price','Test predicted close price'])

plotdf = pd.DataFrame({'date': close_stock['date'],
                       'original_close': close_stock['close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value':'Close price','date': 'Date'})
fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):

    if(len(temp_input)>time_step):

        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)

        yhat = my_model.predict(x_input)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat.tolist())
        temp_input=temp_input[1:]

        lst_output.extend(yhat.tolist())
        i=i+1

    else:
        yhat = my_model.predict(x_input)

        temp_input.extend(yhat.tolist())
        lst_output.extend(yhat.tolist())

        i=i+1

print("Output of predicted next days: ", len(lst_output))

last_days = np.arange(1, time_step + 1)
pred_days = 30  # Define pred_days
day_pred = np.arange(time_step + 1, time_step + pred_days + 1)
print(last_days)
print(day_pred)

temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

new_pred_plot = pd.DataFrame({
    'last_original_days_value':last_original_days_value,
    'next_predicted_days_value':next_predicted_days_value
})

names = cycle(['Last 15 days close price','Predicted next 30 days close price'])
fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                      new_pred_plot['next_predicted_days_value']],
              labels={'value': 'Close price','index': 'Timestamp'})
fig.update_layout(title_text='Compare last 15 days vs next 30 days',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

my_model=closedf.tolist()
my_model.extend((np.array(lst_output).reshape(-1,1)).tolist())
my_model=scaler.inverse_transform(my_model).reshape(1,-1).tolist()[0]

names = cycle(['Close Price'])

fig = px.line(my_model,labels={'value': 'Close price','index': 'Timestamp'})
fig.update_layout(title_text='Plotting whole closing price with prediction',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Cryptocurrency')
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import numpy as np
import math

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def predict_closing_prices(crypto_df):
    # Preprocess the data
    closedf = crypto_df[['date','close']]
    closedf = closedf[closedf['date'] > '2020-09-13']
    close_stock = closedf.copy()
    del closedf['date']

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    closedf = scaler.fit_transform(np.array(closedf).reshape(-1,1))

    # Split the data into training and testing sets
    training_size = int(len(closedf) * 0.70)
    test_size = len(closedf) - training_size
    train_data, test_data = closedf[0:training_size,:], closedf[training_size:len(closedf),:1]

    # Create dataset matrices
    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Build the XGBoost model
    my_model = XGBRegressor(n_estimators=1000)
    my_model.fit(X_train, y_train, verbose=False)

    # Make predictions
    predictions = my_model.predict(X_test)

    # Evaluate the model
    print("Mean Absolute Error - MAE : ", mean_absolute_error(y_test, predictions))
    print("Root Mean Squared Error - RMSE : ", math.sqrt(mean_squared_error(y_test, predictions)))

    # Predict next 30 days closing prices
    x_input = test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    pred_days = 30
    i = 0
    while i < pred_days:
        if len(temp_input) > time_step:
            x_input = np.array(temp_input[1:]).reshape(1,-1)
            yhat = my_model.predict(x_input)
            temp_input.extend(yhat.tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i += 1
        else:
            yhat = my_model.predict(x_input)
            temp_input.extend(yhat.tolist())
            lst_output.extend(yhat.tolist())
            i += 1

    # Inverse scaling
    lst_output = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    return lst_output

# Predict closing prices for Bitcoin
bitcoin_predictions = predict_closing_prices(bitcoindf)

# Display the predictions for Bitcoin
print("Bitcoin Predictions:", bitcoin_predictions)

# Predict closing prices for Dogecoin
dogecoin_predictions = predict_closing_prices(dogecoindf)

# Display the predictions for Dogecoin
print("Dogecoin Predictions:", dogecoin_predictions)

# Predict closing prices for Ethereum
ethereum_predictions = predict_closing_prices(ethereumdf)

# Display the predictions for Ethereum
print("Ethereum Predictions:", ethereum_predictions)

# Predict closing prices for Cardano
cardano_predictions = predict_closing_prices(cardanodf)

# Display the predictions for Cardano
print("Cardano Predictions:", cardano_predictions)

import matplotlib.pyplot as plt

# Predict closing prices for Bitcoin
bitcoin_predictions = predict_closing_prices(bitcoindf)

# Plot Bitcoin Predictions
plt.figure(figsize=(10, 5))
plt.plot(bitcoin_predictions, label='Bitcoin Predictions', color='blue')
plt.title('Bitcoin Predicted Closing Prices')
plt.xlabel('Days')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# Predict closing prices for Dogecoin
dogecoin_predictions = predict_closing_prices(dogecoindf)

# Plot Dogecoin Predictions
plt.figure(figsize=(10, 5))
plt.plot(dogecoin_predictions, label='Dogecoin Predictions', color='green')
plt.title('Dogecoin Predicted Closing Prices')
plt.xlabel('Days')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# Predict closing prices for Ethereum
ethereum_predictions = predict_closing_prices(ethereumdf)

# Plot Ethereum Predictions
plt.figure(figsize=(10, 5))
plt.plot(ethereum_predictions, label='Ethereum Predictions', color='orange')
plt.title('Ethereum Predicted Closing Prices')
plt.xlabel('Days')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# Predict closing prices for Cardano
cardano_predictions = predict_closing_prices(cardanodf)

# Plot Cardano Predictions
plt.figure(figsize=(10, 5))
plt.plot(cardano_predictions, label='Cardano Predictions', color='red')
plt.title('Cardano Predicted Closing Prices')
plt.xlabel('Days')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

import numpy as np

import pickle

# Save the trained model
with open('my_model.pkl', 'wb') as file:
    pickle.dump(my_model, file)

# Save the training dataset
with open('train_dataset.pkl', 'wb') as file:
    pickle.dump(train_data, file)
