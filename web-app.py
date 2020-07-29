import pickle

import pandas as pd
import streamlit as st


from sklearn.linear_model import LinearRegression

st.title(''' Simple Linear Regression application ''')


st.sidebar.header('Inputs for the model.')


algorithm = st.sidebar.selectbox(
    'Algorithm ', ('Linear Regression', 'test1', 'test2'))
st.subheader(algorithm)

st.subheader('Input Parameters')


def input_features():
    stones = st.sidebar.slider('Stone (Measured Weight)', 0.0, 12.0, (5.0))
    pounds = st.sidebar.slider('Pounds (Measured Weight)', 0.0, 13.0, (4.0))
    ounces = st.sidebar.slider('Ounces (Measured Weight)', 0.0, 8.0, (2.0))
    weight_oz = st.sidebar.number_input(
        'Total Weight in Ounces', min_value=1, value=2704)
    cal = st.sidebar.number_input(
        'Calories (One Day)', min_value=1,  value=1400)
    _ = st.sidebar.subheader(f'Calories per Ounce {round(cal/ weight_oz, 2)}')
    cal_per_oz = round(cal / weight_oz, 2)

    five_donuts = st.sidebar.selectbox('five donuts', ('True', 'False'))
    walk = st.sidebar.selectbox('Walk', ('True', 'False'))
    run = st.sidebar.selectbox('Run', ('True', 'False'))
    wine = st.sidebar.selectbox('Wine', ('True', 'False'))

    prot = st.sidebar.selectbox('Prot', ('True', 'False'))
    weight = st.sidebar.selectbox('Weight', ('True', 'False'))
    st.write('')

    data = {
        'Stones': stones,
        'Pounds': pounds,
        'ounces': ounces,
        'weight_oz': weight_oz,
        'cal': cal,
        'cal_per_oz': cal_per_oz,
        'five_donuts': five_donuts,
        'walk': walk,
        'run': run,
        'wine': wine,
        'prot': prot,
        'weight': weight
    }

    return data


df = pd.DataFrame(data=input_features(), index=[1])

df.replace({'True': 1, 'False': 0}, inplace=True)

st.table(df)


# importing model
#stdScaler = pickle.load(open('standardScaler.pk', 'rb'))
model = pickle.load(open('model.pk', 'rb'))

ypred = model.predict(df)

st.write(ypred)
