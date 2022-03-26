import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import time
import tkinter as tk
from tkinter import Label, ttk
import tkcalendar as tkc


from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from  keras.models import Sequential

import urllib3


intervals = ['1d', '5d', '1m', '6m', 'ytd', '1y', '5y', 'max']


# Tkinter box

root = tk.Tk()

box = tk.Canvas(root, width = 270, height = 300)
root.title('Predicción de Precios')
box.pack()
entry1Label = Label(box, text='Introduce el simbolo de la moneda: ').place(x= 5,y =5)
entry1 = tk.Entry(root)
entry1Label = Label(box, text='Introduce el simbolo de la moneda: ').place(x= 5,y =5)
entryDateLabel = Label(box, text='Introduce el periodo de tiempo: ').place(x= 5,y =60)
mindateStart = dt.datetime(2000,1,1)
maxdateStart = dt.datetime.now()
mindateEnd = dt.datetime(2000,1,1)

calStart = tkc.DateEntry(root, selectmode='day', mindate= mindateStart, maxdate = maxdateStart)
calEnd = tkc.DateEntry(root, selectmode= 'day', mindate= mindateStart, maxdate = maxdateStart)
entryDateLabel = Label(box, text='Introduce el intervalo: ').place(x= 5,y =110)
combo = ttk.Combobox(values=intervals, state='readonly')
combo.current(0)

box.create_window(120, 40,window = entry1)
box.create_window(80, 90, window = calStart)
box.create_window(200, 90, window = calEnd)
box.create_window(80, 140,width=60, window=combo)



def getPrediction():

    x1 = entry1.get()
    startDate = calStart.get_date()
    endDate = calEnd.get_date


    #Crypto
    crypto_currency = x1.upper()
    against_currency = 'USD'

    start = dt.datetime(startDate)
    end = dt.datetime(endDate)

    data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', start, end)


    #Preparando los datos

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

    prediction_days = 60
    future_day= 30

    x_train, y_train = [], []

    for x in range(prediction_days , len(scaled_data) - future_day):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x + future_day, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Crear una red Neuronal

    model = Sequential()

    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dense(units = 1))

    model.compile(optimizer = 'rmsprop', loss = 'mse')
    model.fit(x_train, y_train, epochs = 20, batch_size = 32)

    # Probando el modelo

    test_start = dt.datetime(2018, 1, 1)
    test_end = dt.datetime.now()

    test_data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', test_start, test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis = 0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1,1)
    model_inputs = scaler.fit_transform(model_inputs)

    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)

    plt.plot(actual_prices, color='red', label = 'Precios actuales')
    plt.plot(prediction_prices, color='green', label = 'Precios predecidos')
    plt.title(f'Predicción de precios {crypto_currency}')
    plt.xlabel('Tiempo')
    plt.ylabel('Precio')
    plt.legend(loc = 'upper right')
    plt.show()

    # Predicción del siguiente día

    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)


#Descargar documento (Todas las monedas)
def getDocument():

    url = 'https://coincodex.com/apps/coincodex/cache/all_coins.json'
    http = urllib3.PoolManager()
    res = http.request('GET', url)
    data = res.data.decode('utf-8')
    df_json = pd.read_json(data)
    df_json.to_excel(f'./allCoins-{dt.datetime.now().strftime("%d")}-{dt.datetime.now().strftime("%m")}-{dt.datetime.now().strftime("%G")}.xlsx')

#Descargar histórico

def downloadHistorical():
    
    x1 = entry1.get()
    startDate = calStart.get_date()
    endDate = calEnd.get_date()
    interval = combo.get() # 1d, 1m
    
    #Crypto
    crypto_currency = x1.upper()
    against_currency = 'USD'

    period1 = int(time.mktime(startDate.timetuple()))
    period2 = int(time.mktime(endDate.timetuple()))
    training = str(startDate.year + 1)
    validation = str(startDate.year + 2)
    

    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{crypto_currency}-{against_currency}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)
    df.to_csv(f'{crypto_currency}-{against_currency}.csv')


def graficar_predicciones():

    x1 = entry1.get()
    startDate = calStart.get_date()
    endDate = calEnd.get_date()
    interval = combo.get() # 1d, 1m
    
    #Crypto
    crypto_currency = x1.upper()
    against_currency = 'USD'

    period1 = int(time.mktime(startDate.timetuple()))
    period2 = int(time.mktime(endDate.timetuple()))
    training = str(startDate.year + 1)
    validation = str(startDate.year + 2)
    

    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{crypto_currency}-{against_currency}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)
    df.to_csv(f'{crypto_currency}-{against_currency}.csv')

    # Lectura de los datos
    #
    dataset = pd.read_csv(f'{crypto_currency}-{against_currency}.csv', index_col='Date', parse_dates=['Date'])
    dataset.head()

    #
    # Sets de entrenamiento y validación 
    # La LSTM se entrenará con datos de {training} hacia atrás. La validación se hará con datos de {validations} en adelante.
    # En ambos casos sólo se usará el valor más alto de la acción para cada día
    #
    set_entrenamiento = dataset[:training].iloc[:,1:2]
    set_validacion = dataset[validation:].iloc[:,1:2]

    # set_entrenamiento['High'].plot(legend=True)
    # set_validacion['High'].plot(legend=True)
    # plt.legend(['Entrenamiento ' + training, 'Validación ' + validation])
    # plt.show()

    # Normalización del set de entrenamiento
    sc = MinMaxScaler(feature_range=(0,1))
    set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

    # La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida 1 dato (la predicción a
    # partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
    time_step = 60
    X_train = []
    Y_train = []
    m = len(set_entrenamiento_escalado)

    for i in range(time_step,m):
        # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
        X_train.append(set_entrenamiento_escalado[i-time_step:i,0])

        # Y: el siguiente dato
        Y_train.append(set_entrenamiento_escalado[i,0])
    X_train, Y_train = np.array(X_train), np.array(Y_train)

    # Reshape X_train para que se ajuste al modelo en Keras
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    #
    # Red LSTM
    #
    dim_entrada = (X_train.shape[1],1)
    dim_salida = 1
    na = 50

    modelo = Sequential()
    modelo.add(LSTM(units=na, input_shape=dim_entrada))
    modelo.add(Dense(units=dim_salida))
    modelo.compile(optimizer='rmsprop', loss='mse')
    modelo.fit(X_train,Y_train,epochs=20,batch_size=32)


    #
    # Validación (predicción del valor de las acciones)
    #
    x_test = set_validacion.values
    x_test = sc.transform(x_test)

    X_test = []
    for i in range(time_step,len(x_test)):
        X_test.append(x_test[i-time_step:i,0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    prediccion = modelo.predict(X_test)
    prediccion = sc.inverse_transform(prediccion)


    # figure1 = plt.Figure(figsize=(6,5), dpi=100)
    # ax1 = figure1.add_subplot(111)
    # bar1 = FigureCanvasTkAgg(figure1, root)
    # bar1.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
    # df1 = df1['Predicción']
    # df1.plot(kind='Valor real de la acción', legend= True, ax=ax1)
    

    plt.plot(set_validacion.values[0:len(prediccion)],color='red', label='Valor real de la acción')
    plt.plot(prediccion, color='blue', label=f'Predicción de la acción {crypto_currency}')
    plt.ylim(1.1 * np.min(prediccion)/2, 1.1 * np.max(prediccion))
    plt.xlabel('Tiempo')
    plt.ylabel('Valor de la acción')
    plt.legend()
    plt.show()

def chartPredict():
    x1 = entry1.get()
    startDate = calStart.get_date()
    endDate = calEnd.get_date()
    interval = combo.get() # 1d, 1m
    
    #Crypto
    crypto_currency = x1.upper()
    against_currency = 'USD'

    period1 = int(time.mktime(startDate.timetuple()))
    period2 = int(time.mktime(endDate.timetuple()))
    training = str(startDate.year + 1)
    validation = str(startDate.year + 2)
    

    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{crypto_currency}-{against_currency}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)
    df.to_csv(f'{crypto_currency}-{against_currency}.csv')

    # Lectura de los datos
    #
    dataset = pd.read_csv(f'{crypto_currency}-{against_currency}.csv', index_col='Date', parse_dates=['Date'])
    dataset.head()

    #
    # Sets de entrenamiento y validación 
    # La LSTM se entrenará con datos de {training} hacia atrás. La validación se hará con datos de {validations} en adelante.
    # En ambos casos sólo se usará el valor más alto de la acción para cada día
    #
    set_entrenamiento = dataset[:training].iloc[:,1:3]
    set_validacion = dataset[validation:].iloc[:,1:3]

    set_entrenamiento['High'].plot(legend=True)
    set_validacion['High'].plot(legend=True)
    plt.legend(['Entrenamiento ' + training, 'Validación ' + validation])
    plt.show()



button1 = tk.Button(text = 'Ver predicción', command = getPrediction)
button2 = tk.Button(text = 'Generar documento', command = getDocument)
button3 = tk.Button(text = 'Descargar histórico', command = downloadHistorical)
button4 = tk.Button(text = 'Graficar predicción histórico', command = graficar_predicciones)
button5 = tk.Button(text = 'Ver gráfica', command = chartPredict)
# box.create_window(120, 50, window = button1)
box.create_window(120, 180, window = button2)
box.create_window(120, 210, window = button3)
box.create_window(120, 240, window = button4)
box.create_window(120, 270, window = button5)


root.mainloop()