import joblib
import pandas as pd
import pickle
import sklearn
import numpy as np
import streamlit as st


# df = pickle.load(open('LPTP_DF.pkl', 'rb'))
df = joblib.load('LPTP_DF.pkl')
model = joblib.load('LPTP.pkl')
# Index(['company', 'inches', 'new_screenresolution', 'new_clockspeed',
# 'new_cpu', 'new_ram', 'HDD', 'SSD', 'new_GPU', 'OS', 'new_weight'],
# dtype='object')


st.title("Laptop Price Predictor")

# brand
company = st.selectbox('Brand', df['company'].unique(), index=4)

# inches = st.number_input('Screen size in inches ', min_value=11, max_value=30,steps=1, format("%.f")  )


inches = st.number_input(
    "Screen size in inches",
    min_value=5.0,
    max_value=30.0,
    step=1.0,
    value=15.6,
    format="%f")

# Ram
new_ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)
# weight
new_weight = st.number_input('Weight of the Laptop', step=0.5, format="%.f", value=2.5)
# clockspeed
new_clockspeed = st.number_input('Clock Speed Laptop', step=0.5, format="%.f", value=2.5)
# screen size
new_screenresolution = st.selectbox("Screen Size", df['new_screenreso'].unique(), index=2)
# cpu
new_cpu = st.selectbox('CPU', [i.capitalize() for i in list(df['new_cpu'].unique())])
# hdd
HDD = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
# sdd
SSD = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
# GPU
new_GPU = st.selectbox('GPU', df['new_gpu'].unique())
# os
OS = st.selectbox('OS', df['new_OS'].unique(), index=2)

tchscrn = st.selectbox('TouchScreen', ["Yes", "No"], index=1)

if tchscrn == "Yes":
    tt = 1
else:
    tt = 0

data = [company, inches, new_ram, new_weight, new_screenresolution, new_cpu, new_clockspeed, new_GPU, OS, tt, HDD, SSD]

predict_df = pd.DataFrame(data=[data], columns=df.columns)


if st.button('Predict Price'):
    result = model.predict(predict_df)
    st.title("Predicted Price is  " + str(int(np.exp(result))))
    del predict_df
