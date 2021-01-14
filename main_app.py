import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes, load_boston
from sklearn import metrics

# Page Layout
st.set_page_config(page_title='Random Forest Regressor App', layout='wide')

st.write("""
    # Random Forest Regressor App
    This aplication use Random Forest Regressor for build regression model using Random Forest algorithm.

    You can try to adjust the hyperparameters to find the best parameters for your data.
""")

# Sidebar
with st.sidebar.header('1.  Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader(
        "Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
        [Example Data](https://raw.githubusercontent.com/GigasTaufan/Random-Forest-Regression-Modelling/main/delaney_solubility_with_descriptors.csv)
    """)
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider(
        'Data split ratio (% for training set', 10, 90, 80, 5)

with st.sidebar.header('2.1. Learning Parameters'):
    parameter_n_estimator = st.sidebar.slider(
        'Number of estimator (n_estimator)', 0, 1000, 100, 100)
    parameter_max_feature = st.sidebar.select_slider(
        'Max Features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_sample_split = st.sidebar.slider(
        'Minimum number of samples required to split an internal node (min_sample_split)', 1, 10, 2, 1)
    parameter_min_sample_leaf = st.sidebar.slider(
        'Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 20, 2, 1)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider(
        'Seed number (random_state)', 0, 1000, 42, 1)

    parameter_criterion = st.sidebar.select_slider(
        'Performance measure (criterion)', options=['mse', 'mae'])
    if (parameter_criterion == 'mse'):
        nama_error = 'MSE'
        error = metrics.mean_squared_error
    else:
        nama_error = 'MAE'
        error = metrics.mean_absolute_error

    parameter_bootstrap = st.sidebar.select_slider(
        'Bootstrap sampels when building trees (bootstrap)', options=[True, False])
    parameter_oob_store = st.sidebar.select_slider(
        'Wether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider(
        'Number of jobs to run in parallel (n_jobs)', options=[1, -1])

# build machine learning model
def build_model(df):
    X = df.iloc[:, :-1]  # using all columns except for the last column as X
    y = df.iloc[:, -1]  # select the last column as y

    st.markdown('**1.2. Data Splits**')
    st.write('Training Set')
    st.info(X.shape)
    st.write('Testing Set')
    st.info(y.shape)

    st.markdown('**1.3. Variable Details:**')
    st.write('X Variables')
    st.info(list(X.columns))
    st.write('Y Variables')
    st.info(y.name)

   # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_size)

    rfr = RandomForestRegressor(n_estimators=parameter_n_estimator,
                                random_state=parameter_random_state,
                                max_features=parameter_max_feature,
                                criterion=parameter_criterion,
                                min_samples_split=parameter_min_sample_split,
                                min_samples_leaf=parameter_min_sample_leaf,
                                bootstrap=parameter_bootstrap,
                                oob_score=parameter_oob_store,
                                n_jobs=parameter_n_jobs)
    rfr.fit(X_train, y_train)

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training Set**')
    y_train_pred = rfr.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info(metrics.r2_score(y_train, y_train_pred))

    st.write('Error (MSE or MAE):')
    st.write(nama_error)
    st.info(error(y_train, y_train_pred))

    st.markdown('**2.2. Testing Set**')
    y_test_pred = rfr.predict(X_test)
    st.write(nama_error)
    st.info(error(y_test, y_test_pred))

    st.subheader('3. Model Parameters')
    st.write(rfr.get_params()) # get the parameters of the model


# Main Display
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of Dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example dataset'):
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        y = pd.Series(boston.target, name='response')
        df = pd.concat([X, y], axis=1)

        st.markdown('The Boston housing dataset is used as the example.')
        st.write(df.head(5))

        build_model(df)
