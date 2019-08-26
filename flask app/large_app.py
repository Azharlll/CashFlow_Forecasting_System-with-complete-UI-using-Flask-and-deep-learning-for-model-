import os
import glob, os, os.path
import numpy as np
import pandas as pd
np.random.seed(101)
from numpy.random import randint, randn
import datetime
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
#import matplotlib.mlab as mlab
#import plotly.plotly as py
#py.sign_in('DemoAccount', 'lr1c37zw81')
import io
import base64
import pygal
import datetime
from datetime import date,timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt, mpld3
CashFLow = pd.read_excel("CashFLow.xlsx")  
import keras                

#import plotly as py
#import plotly.plotly as py
#import plotly.tools as plotly_tools
#import plotly.graph_objs as go
# with open('Expenses_Amt_model.sav', 'rb') as f1:
#     Exp_pickle = pickle.load(f1)

# with open('Payable_Amt_model.sav', 'rb') as f2:
#     payable_pickle = pickle.load(f2)

# with open('Receivable_Amt_model.sav', 'rb') as f3:
#     receivable_pickle = pickle.load(f3)

from flask import Flask, render_template, request
#from gevent.pywsgi import WSGIServer

__author__ = 'Azhar'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGE_FOLDER = os.path.join('static', 'pics')
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ndx")
def ndx():
    return render_template("index.html")    

print('poppp')

@app.route("/Receivable", methods=['POST'])
def Receivable():
    return render_template('form.html',vari='Receivable Amount Forecast')

@app.route("/Payable", methods=['POST'])
def Payable():
    return render_template('form.html',vari='Payable Amount Forecast')

@app.route("/Expenses", methods=['POST'])    
def Expenses():
    return render_template('form.html',vari='Expenses Amount Forecast')

@app.route("/NetProfit", methods=['POST'])    
def NetProfit():
    return render_template('form.html',vari='Net Profit Forecast')    

@app.route("/prediction", methods=['POST'])
def prediction():
    filelist = glob.glob(os.path.join('static/', "*.png"))
        #print(filelist)
    #regressor= tf.Variable(True, use_resource=True)
    keras.backend.clear_session()

    for f in filelist:
        os.remove(f)
        plt.savefig('static/new1_plot.png')
    print('babu')    
   
    dtt1 = (request.form['date1']) 
    dtt2 = (request.form['date2'])
    print(dtt1,dtt2)
    dt1=int(dtt1[:4])
    dt2=int(dtt1[5:7])
    dt3=int(dtt1[8:10])
    dt4=int(dtt2[:4])
    dt5=int(dtt2[5:7])
    dt6=int(dtt2[8:10])
    print(dt1,dt2,dt3,dt4,dt5,dt6)
    #dt3 = int(request.form['date3'])
    #dt4 = int(request.form['date4'])
    #dt5 = int(request.form['date5'])
    #dt6 = int(request.form['date6'])


    print(dt1,dt2)
    print(type(dt1))
    print(request.form)
    if 'Net Profit Forecast' in request.form:
        data='Net_Profit'
        print('In net_pro')
        
        regressor=pickle.load(open('Net_Profit_model.sav','rb'))
        print('In2 net_pro')

    if 'Receivable Amount Forecast' in request.form:
        print('hi')
        data = 'Receivable_Amt'
        regressor = pickle.load(open('Receivable_Amt_model.sav','rb'))
        #print(regressor)

    if 'Payable Amount Forecast' in request.form:
        print('hi2')
        data = 'Payable_Amt'
        regressor = pickle.load(open('Payable_Amt_model.sav','rb'))

    if  'Expenses Amount Forecast' in request.form:
        data = 'Expenses_Amt'
        regressor = pickle.load(open('Expenses_Amt_model.sav','rb'))

    print(data)

    trn_indx,trn2_indx=1258,1108 
    def model_func(data):
        dataset_total=np.array(CashFLow[data].values)
        #X_train1, X_test = train_test_split( CashFLow['Payable_Amt'],test_size=0.33, random_state=42)
        dataset_test=dataset_total[trn2_indx:]
        training_set=dataset_total[:trn2_indx]
        # Feature Scaling
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range = (0, 1))
        training_set_scaled = sc.fit_transform(training_set.reshape(-1,1))

        # Creating a data structure with 60 timesteps and 1 output
        X_train = []
        y_train = []
        for i in range(60, trn2_indx):
            X_train.append(training_set_scaled[i-60:i, 0])
            y_train.append(training_set_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Reshaping
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        return dataset_test, dataset_total,training_set,sc,data   



    def plot_func(dataset_test,dataset_total,training_set,regressor,sc):
    
        real_stock_price = dataset_test
        #print(len(dataset_test))
        # Getting the predicted stock price of 2017
        #dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
        inputs = dataset_total[len(training_set) - 60:]
        #print(len(training_set), len(training_set) - 60,len(dataset_total))
        #print('inputs',inputs)
        inputs = inputs.reshape(-1,1)
        inputs = sc.transform(inputs)
        X_test = []
        for i in range(60, len(dataset_test)+60):
            X_test.append(inputs[i-60:i, 0])
            #print(i-60, i)
            #print(X_test,'\n')
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        #print(X_test.shape)
        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        #print(len(predicted_stock_price))
        # Visualising the results

        img = io.BytesIO()
        plt.figure(figsize=(20,4))
        plt.subplots()
        plt.plot(real_stock_price, color = 'red', label = 'Real Price')
        plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Price')
        plt.title(' Prediction')
        plt.xlabel('Time')
        plt.ylabel(data+'Price')
        plt.legend()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return 'data:image/png;base64,{}'.format(graph_url)

        #img = io.BytesIO()
        #plt.plot(x_coordinates, y_coordinates)
        #plt.savefig(img, format='png')
        #img.seek(0)

        #plot_url = base64.b64encode(img.getvalue()).decode()

        #return '<img src="data:image/png;base64,{}">'.format(plot_url)
        
        ''' filelist = glob.glob(os.path.join('static/', "*.png"))
        #print(filelist)
        for f in filelist:
            os.remove(f)
        
        #plt.show()
        plt.savefig('static/new1_plot.png')
        #mpld3.show()


        graph = pygal.Line()       
        graph.title = '% Change Coolness of programming languages over time.'
        #graph.x_labels =  
        #graph.add('Python',  [15, 31, 89, 200, 356, 900])
        #graph.add('Java',real_stock_price)
        graph.add('C++', predicted_stock_price)
        graph.add('All others combined!',real_stock_price )
        graph_data = graph.render_data_uri()
        return graph_data'''  
        


    def user_inp(df,regressor,sc):
        #reg = regressor
        print("Enter the Date range for forecasting")
        #year = int(input('Enter'))
        #month = int(input('Enter a month'))
        #day = int(input('Enter a day'))
        date1 = datetime.date(dt1, dt2, dt2)
        #print(str(date1),type(date1))
        #year = int(input('Enter a year'))
        #month = int(input('Enter a month'))
        #day = int(input('Enter a day'))
        date2 = datetime.date(dt4, dt5, dt6)

        delt=date2-date1
        #print(df)
        for i in range(len(df.index)):
            if str(df.index[i])[:10] == str(date1):
                dt_inx=i
                print('yo man')
                #print(i)
                break

        ts=training_set
        pred_List=[]
        for i in range(delt.days):
            #print(len(ts))
            t_test=ts[(dt_inx+i)-60:(dt_inx+i)]
            t_test = t_test.reshape(-1,1)
            t_test = sc.transform(t_test)
            #print(len(t_test),t_test)
            t_test = np.array(t_test)
            
            #t_test = np.reshape(t_test, (t_test.shape[0], t_test.shape[1], 1))
            t_test = np.reshape(t_test, (1, t_test.shape[0], t_test.shape[1]))
            #print(t_test)
            predicted_stock_price = regressor.predict(t_test)
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)
            #print(predicted_stock_price,Amount[dt_inx+i])
            ts=np.append(ts,predicted_stock_price)
            pred_List.append(predicted_stock_price)
            #print(len(ts))
            #############################################
            

        ts=[]  
        return pred_List,date1,date2,dt_inx,delt


    def plot_Predic(pred_List,date1,date2,dt_inx,delt,data):
        Date2=pd.date_range(start=date1,end=date2-timedelta(days=1))
        df=pd.DataFrame(dict(Date2=Date2,pred_l=pred_List),index=Date2)

        img = io.BytesIO()

        
        ##############################################
        #xy_data = go.Scatter( x=pred_List, y=Date2, mode='markers', marker=dict(size=4), name='AAPL' )
        #data = [xy_data]

        #py.iplot(data, filename='apple stock moving average')
        ##############################################################
        plt.figure(figsize=(10,4))
        plt.plot(df['pred_l'])
        #plt.plot(CashFLow[data][dt_inx:dt_inx+delt.days])
        plt.title(' Prediction')
        plt.xlabel('Time')
        plt.ylabel(data+' Amount')
        plt.legend()  
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return 'data:image/png;base64,{}'.format(graph_url),Date2


        #mpld3.show() 
        #import glob, os, os.path


        '''
        graph = pygal.Line(show_x_labels=True,show_y_labels=True)       
        graph.title = '% Change Coolness of programming languages over time.'
        graph.x_labels = Date2
        #graph.x_labels=df['pred_l']
        graph.add('All others combined!',CashFLow[data][dt_inx:dt_inx+delt.days] )
        graph.add('C++', df['pred_l'])
        
        graph_data = graph.render_data_uri()
        return graph_data,Date2'''
      
        # filelist = glob.glob(os.path.join('static/', "*.png"))
        # #print(filelist)
        # for f in filelist:
        #     os.remove(f)'''
        #plt.savefig('static/new1_plot.png')
        #return df['pred_l'],(CashFLow[data][dt_inx:dt_inx+delt.days]
        #return Date2

    dataset_test,dataset_total,training_set,sc,data=model_func(data)
    #print(dataset_test,dataset_total,training_set,sc,data)
    graph_url=plot_func(dataset_test,dataset_total,training_set,regressor,sc)
    pred_List,date1,date2,dt_inx,delt=user_inp(CashFLow,regressor,sc)
    graph_url2,Date2=plot_Predic(pred_List,date1,date2,dt_inx,delt,data)
    #print(pred_List)
    dt_list = []
    dtf = pd.DataFrame()
    
    for i in pred_List:
        dt_list.append(i[0][0])
    list1=[1,2,3,4,5]
    dtf[data] = dt_list
    dtf['Date']=Date2
    #print(dtf)
    
    # dpd=dtf[[data]
    # dpd['month'] = dtf['Date'].apply(lambda l : str(l)[:7])
    # dpd.groupby('month').sum()
    # print(dpd)
    
    '''@app.route("/upload/<filname>")
    def send_image(filname):
    return send_from_directory("images",filname)'''

    #return render_template('complete2.html')
    

    return render_template('complete.html',table = dtf, name = data, graph1=graph_url,graph2=graph_url2)





if __name__ == "__main__":
    app.run(port=5000, debug=False)
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
