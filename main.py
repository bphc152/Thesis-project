                    ###---import library
import streamlit as st
import streamlit.components.v1 as stc 
import pandas as pd
import numpy as np
import os
import pickle
import time

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor,VotingRegressor,StackingRegressor
from sklearn.linear_model import Ridge, RidgeCV,Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



                    ###---main

##---load csv file
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
df_train = pd.read_csv('./hp_train.csv')
df_test = pd.read_csv('./hp_test.csv')



def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
  

##---load model
with open('lgb.pkl','rb') as lgb_pkl:
  lgb = pickle.load(lgb_pkl)

with open('xgb.pkl','rb') as xgb_pkl:
  xgb = pickle.load(xgb_pkl)

with open('gbr.pkl','rb') as gbr_pkl:
  gbr = pickle.load(gbr_pkl)

with open('svr.pkl','rb') as svr_pkl:
  svr = pickle.load(svr_pkl)

with open('ridge.pkl','rb') as ridge_pkl:
  ridge = pickle.load(ridge_pkl)

with open('enet.pkl','rb') as enet_pkl:
  enet = pickle.load(enet_pkl)

with open('voting.pkl','rb') as voting_pkl:
  voting = pickle.load(voting_pkl)
  
with open('stacking.pkl','rb') as stacking_pkl:
  stacking = pickle.load(stacking_pkl)

with open('rf.pkl','rb') as rf_pkl:
  rf = pickle.load(rf_pkl)



#---Title tag web tab
st.set_page_config(page_icon="🏠", page_title="House Price Predictor")


##---sidebar
st.sidebar.header('Choose Features')
page = st.sidebar.radio('Menu', ['Main Page', 'EDA', 'ML','About'])

#--- feature input
house_id = st.sidebar.selectbox("House ID",test["Id"])

data = test[test['Id']==house_id]
df_data = df_test[df_test['Id']==house_id]
df_data = df_data.drop(['Id'], axis=1)

train_labels = train['SalePrice'].reset_index(drop=True)

###--------------
if page == 'Main Page':
  html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center; font-family:sans-serif">House Price Predictor</h1>
		<h4 style="color:white;text-align:center;"></h4>
		</div>
		"""
  
  st.title('Main Page')
  stc.html(html_temp)
  st.header(':wave: Hi, welcome to house prices predictor')
  st.write('Select house ID to see feature detail')
  st.subheader('Details of house you chose')
  st.write(data)


###--------------
if page =='EDA':
  st.title('EDA')
  st.image('./edabanner.png')
  st.write('This page will show you some feature affect to price of the house')
  st.header('Correlation')
  
  
  cols = train[['Id','SalePrice','OverallQual','1stFlrSF','2ndFlrSF','GrLivArea','FullBath','HalfBath','TotRmsAbvGrd','GarageCars','GarageArea']]
  dfcol = cols[cols['Id']==house_id]
  corrcol = cols.corr()
  fig, ax = plt.subplots(figsize=(12, 9))
  sns.heatmap(corrcol, vmax=.8,square=True,xticklabels=True, yticklabels=True,cmap="Blues")
  corclick = st.button('Show Correlation')
  if corclick:
    st.button('Hide')
    st.write('Show you correlation between important characteristics and house price')
    st.write(fig)
  # col1
  # with col1:
    
    
    


###--------------
if page == 'ML':
  st.title('Machine Learning')
  st.image('./housesbanner.png')
  # st.header('You can select model to observe price of house')
  st.subheader('Please select model you want to predict')
  model_box = st.selectbox(' Model', list(['Light Gradient Boost',
                                    'Extreme Gradient Boost',
                                    'Gradient Boost',
                                    'Random Forest Regressor',
                                    'Ridge Regression',
                                    'Lasso Regression',
                                    'Ensemble Voting Regressor',
                                    'Ensemble Stacking Regressor']))
  if model_box == 'Light Gradient Boost':
    st.header('You chose Light Gradient Boost for prediction')
    click1 = st.button('Predict')
    pred_lgb = lgb.predict(df_data)
    print(f'light Boost: {pred_lgb}\n')
    pred_lgb=np.expm1(pred_lgb) 
    df_pred=pd.DataFrame({'SalePrice':pred_lgb}).reset_index(drop=True)
    
    if click1:
      with st.spinner(f'Predicting with {model_box} ... Please wait'):
        time.sleep(3)
        st.success('successfully')
        st.write(df_pred, 'Do you have enough money to buy this house?')

  
      
  elif model_box == 'Extreme Gradient Boost':
    st.header('You chose Extreme Gradient Boost for prediction')
    click1 = st.button('Predict')
    pred_xgb = xgb.predict(df_data)
    print(f'ex Boost: {pred_xgb}\n')
    pred_xgb=np.expm1(pred_xgb)  
    df_pred=pd.DataFrame({'SalePrice':pred_xgb}).reset_index(drop=True)
  
    if click1:
      with st.spinner(f'Predicting with {model_box} ... Please wait'):
        time.sleep(3)
        st.success('successfully')
        st.write(df_pred, 'Do you have enough money to buy this house?')


  if model_box == 'Gradient Boost':
    st.header('You chose Gradient Boost for prediction')
    click1 = st.button('Predict')
    pred_gbr = gbr.predict(df_data) 
    print(f'Gradient Boost: {pred_gbr}\n')
    pred_gbr=np.expm1(pred_gbr)
    df_pred=pd.DataFrame({'SalePrice':pred_gbr}).reset_index(drop=True)
  
    if click1:
      with st.spinner(f'Predicting with {model_box} ... Please wait'):
        time.sleep(3)
        st.success('successfully')
        st.write(df_pred, 'Do you have enough money to buy this house?')


  if model_box == 'Random Forest Regressor':
    st.header('You chose Random Forest Regressor for prediction')
    click1 = st.button('Predict')
    pred_rf = rf.predict(df_data) 
    print(f'Support Vector Regression: {pred_rf}\n')
    pred_rf=np.expm1(pred_rf)
    df_pred=pd.DataFrame({'SalePrice':pred_rf}).reset_index(drop=True)

    if click1:
      with st.spinner(f'Predicting with {model_box} ... Please wait'):
        time.sleep(3)
        st.success('successfully')
        st.write(df_pred, 'Do you have enough money to buy this house?')


  if model_box == 'Ridge Regression':
    st.header('You chose Ridge Regression for prediction')
    click1 = st.button('Predict')
    pred_ridge = lgb.predict(df_data)
    print(f'Ridge: {pred_ridge}\n')
    pred_ridge=np.expm1(pred_ridge)
    df_pred=pd.DataFrame({'SalePrice':pred_ridge}).reset_index(drop=True)

    if click1:
      with st.spinner(f'Predicting with {model_box} ... Please wait'):
        time.sleep(3)
        st.success('successfully')
        st.write(df_pred, 'Do you have enough money to buy this house?')


  if model_box == 'Lasso Regression':
    st.header('You chose :Lasso Regression for prediction')
    click1 = st.button('Predict')
    pred_lasso = lgb.predict(df_data) 
    print(f'Lasso: {pred_lasso}\n')
    pred_lasso=np.expm1(pred_lasso)
    df_pred=pd.DataFrame({'SalePrice':pred_lasso}).reset_index(drop=True)

    if click1:
      with st.spinner(f'Predicting with {model_box} ... Please wait'):
        time.sleep(3)
        st.success('successfully')
        st.write(df_pred, 'Do you have enough money to buy this house?')


  if model_box == 'Ensemble Voting Regressor':
    st.header('You chose Ensemble Voting Regressor for prediction')
    click1 = st.button('Predict')
    pred_voting = lgb.predict(df_data) 
    print(f'voting: {pred_voting}\n')
    pred_voting=np.expm1(pred_voting)
    df_pred=pd.DataFrame({'SalePrice':pred_voting}).reset_index(drop=True)
    
    if click1:
      with st.spinner(f'Predicting with {model_box} ... Please wait'):
        time.sleep(3)
        st.success('successfully')
        st.write(df_pred, 'Do you have enough money to buy this house?')


  if model_box == 'Ensemble Stacking Regressor':
    st.header('You chose Ensemble Stacking Regressor for prediction')
    click1 = st.button('Predict')
    pred_stacking = lgb.predict(df_data) 
    print(f'stacking: {pred_stacking}\n')
    pred_stacking=np.expm1(pred_stacking)
    df_pred=pd.DataFrame({'SalePrice':pred_stacking}).reset_index(drop=True)
    
    if click1:
      with st.spinner(f'Predicting with {model_box} ... Please wait'):
        time.sleep(3)
        st.success('successfully')
        st.write(df_pred, 'Do you have enough money to buy this house?')
###--------------
if page == 'About':
  html_temp2 = '''
  <link rel="stylesheet" href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'>
  <i style="font-size:32px" class='fa fa-github fa-spin' ></i>
  <span ><a style="font-size:30px" href="github.com/bphc152"> /bphc152</a></span>
  '''
  
  st.title('About')
  st.markdown('_Made by me using streamlit_')
  st.subheader('👨‍🎓 Le Bao Phuc')
  st.subheader(':id: ITDSIU18033')
  st.subheader(':email: phuclb152@gmail.com')
  stc.html(html_temp2)
  
  
  #responsive on different types of devices
  
