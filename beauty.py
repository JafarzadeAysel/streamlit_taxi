import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

@st.cache
def get_data(filename):
    data = pd.read_parquet(filename)
    return data
with header:
    st.title('welcome to my project')
    st.text('In this project I look into the transactions of taxis in NYC..')


with dataset:
    st.header('NYC taxi dataset')
    st.text('I found this dataset on Kaggle')

    text_data = get_data('data/green_tripdata_2022.parquet') 
    st.write(text_data.head())
    
    st.subheader('VendorID distribution on NYC dataset')
    vendors = pd.DataFrame(text_data['VendorID'].value_counts())
    st.bar_chart(vendors)
    

with features:
    st.header('Features I created')
    st.markdown('* **First feature:** I created this feature because of this... I calculated it using this logic...')
    st.markdown('* **Second feature**: I created this second feature because of ....')

with model_training:
    st.header('Time to train model')
    st.text('Here you get to choose the hyperparameters of the model and see how the performance changes')

    sel_col,disp_col = st.columns(2)

    max_depth = sel_col.slider('How much max_depth you choose?',min_value = 10, max_value = 100, value = 20,step = 10)
    n_estimators = sel_col.selectbox("How many trees should there be?", options= [100,200,300, 'No Limit'], index = 0)

    input_feature = sel_col.text_input('which feature should be input feature?', 'PULocationID')
    
    sel_col.text('Here is list of columns')
    sel_col.write(text_data.columns)

    if n_estimators == 'No Limit':
        regr = RandomForestRegressor(max_depth = max_depth)
    else:
        regr = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators)

    X = text_data[[input_feature]]
    y = text_data[['trip_distance']]

    regr.fit(X.values,y.values)
    prediction = regr.predict(y)

    disp_col.subheader('Mean absolute error of model is ')
    disp_col.write(mean_absolute_error(y,prediction))

    disp_col.subheader('Mean squared error of the model is ')
    disp_col.write(mean_squared_error(y,prediction))

    disp_col.subheader('R square score of the model is ')
    disp_col.write(r2_score(y,prediction))
