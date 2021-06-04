from numpy import random
import streamlit as st
import numpy as np
import pandas as pd
import time
from data_cleaner import eliza_cleaning, eliza_fillna
from pickle import load
import xgboost as xgb
from xgboost import XGBRegressor


@st.cache
def load_data():
    raw_datas = pd.read_csv('https://raw.githubusercontent.com/JulienAlardot/challenge-collecting-data/main/Data/database.csv')
    datas = eliza_cleaning(raw_datas)
    datas = eliza_fillna(datas)
    return datas

@st.cache
def get_median_price(post_code):
    median = pd.read_csv('median.csv')
    post = pd.read_csv('post_codes.csv', sep=';')
    median['Gemeente'] = median['Gemeente'].str.lower()
    post['Commune Principale'] = post['Commune principale'].str.lower()
    median_with_post = median.merge(post[['Code postal', 'Commune Principale']], how='left', left_on='Gemeente', right_on='Commune Principale')
    median_with_post = median_with_post.groupby('Gemeente').median()
    median_with_post['Mediaanprijs 2020'].fillna(median_with_post['Mediaanprijs 2019'], inplace=True)
    median_with_post['Mediaanprijs 2020'].fillna(median_with_post['Mediaanprijs 2018'], inplace=True)
    post_with_median = post[['Code postal','Commune principale']].merge(median_with_post[['Code postal', 'Mediaanprijs 2020']], how='left', left_on='Code postal', right_on='Code postal')
    post_with_median.sort_values(by='Code postal', inplace=True)
    post_with_median.fillna(method='bfill', inplace=True)
    post_with_median.fillna(method='ffill', inplace=True)
    post_with_median.drop_duplicates(inplace=True)
    post_with_median.set_index('Code postal', drop=True, inplace=True)
    post_with_median.pop('Commune principale')
    try:
        post_code = int(post_code)
        return post_with_median.loc[post_code, 'Mediaanprijs 2020']
    except:
        return post_with_median.mean()['Mediaanprijs 2020']

def fill(data):
    if not garden:
        data.loc[0,'garden_area'] = 0
    elif data.loc[0,'garden_area']== "":
        data.loc[0,'garden_area'] = np.nan
    else:
        data.loc[0,'garden_area'] = float(data.loc[0,'garden_area'])
    if not terrace:
        data.loc[0,'terrace_area'] =0
    elif data.loc[0,'terrace_area']== "":
        data.loc[0,'terrace_area'] = np.nan
    else:
        data.loc[0,'terrace_area'] = float(data.loc[0,'terrace_area'])
    if data.loc[0,'surface_of_the_land'] == "":
        data.loc[0,'surface_of_the_land'] = np.nan
    else:
        data.loc[0,'surface_of_the_land'] = float(data.loc[0,'surface_of_the_land'])
    if data.loc[0,'number_of_rooms'] == "":
        data.loc[0,'number_of_rooms'] = np.nan
    else:
        data.loc[0,'number_of_rooms'] = int(data.loc[0,'number_of_rooms'])
    if data.loc[0,'number_of_facades'] == "":
        data.loc[0,'number_of_facades'] = np.nan
    else:
        data.loc[0,'number_of_facades'] = int(data.loc[0,'number_of_facades'])
    if data.loc[0,'area'] == "":
        data.loc[0,'area'] = np.nan
    else:
        data.loc[0,'area'] = float(data.loc[0,'area'])
    if data.loc[0,'surface_of_the_land'] == "":
        data.loc[0,'surface_of_the_land'] = np.nan
    else:
        data.loc[0,'surface_of_the_land'] = float(data.loc[0,'surface_of_the_land'])
    return data



st.title('Price Prediction for housing in Belgium')

type_of_property = st.selectbox('Type', ['house', 'apartment'])


provinces_flandres = ['Antwerp', 'West-Vlanderen', 'Oost-Vlanderen', 'Vlaams-Brabant', 'Limburg']
provinces_wallonie = ['Hainaut', 'Liège', 'Namur', 'Luxembourg', 'Brabant Wallon']
province_brussel = ['Brussels']
provinces = province_brussel+provinces_flandres+provinces_wallonie
loc_column, prov_column = st.beta_columns(2)
with loc_column:
    loc = st.text_input('Postal Code', 1000)
with prov_column:
    prov = st.selectbox('Province', provinces)


state = st.selectbox('State of the Building', ['good', 'new', 'to renovate'])

room_col, facade_col = st.beta_columns(2)
with room_col:
    room = st.text_input('Number of Rooms')
with facade_col:
    facade = st.text_input('Number of Facades')

area_col, area_land_col = st.beta_columns(2)
with area_col:
    area = st.text_input('Area')
with area_land_col:
    area_land = st.text_input('Area of the Land')


garden_area=np.nan
garden_column, garden_area_column = st.beta_columns(2)
with garden_column:
    garden = st.checkbox('Garden')
if garden:
    with garden_area_column:
        garden_area = st.text_input('Area of the Garden')    


terrace_column, terrace_area_column = st.beta_columns(2)
terrace_area =  np.nan
with terrace_column:
    terrace = st.checkbox('Terrace')
if terrace:
    with terrace_area_column:
        terrace_area = st.text_input('Area of the Terrace')   
        


kitchen_col, furnished_col, fire_col, pool_col = st.beta_columns(4)
with kitchen_col:
    kitchen = st.checkbox('Fully equipped kitchen', key='kitchen')
with furnished_col:
    furnished = st.checkbox('Furnished', key='furnished')
with fire_col:
    fire = st.checkbox('Open Fire', key='fire')
with pool_col:
    pool = st.checkbox('Swimming Pool', key='pool')

@st.cache
def convert_province(prov):
    if prov in provinces_wallonie:
        return 'Wallonie'
    elif prov in provinces_flandres:
        return 'Vlaams'
    elif prov == 'Brussels':
        return 'Brussels Capital'
    else:
        return '__________'




building = {'median_price':[get_median_price(loc)], 
            'locality':[int(loc)],
            'number_of_rooms':[room],
            'area':[area],
            'fully_equipped_kitchen':[int(kitchen)],
            'furnished':[int(furnished)],
            'open_fire':[int(fire)],
            "terrace":[int(terrace)],
            'terrace_area':[terrace_area],
            'garden':[int(garden)],
            'garden_area':[garden_area],
            'surface_of_the_land':[area_land],
            "number_of_facades":[facade],
            'swimming_pool':[int(pool)],
            'state_of_the_building':[state],
            'province':[prov],
            'region': [convert_province(prov)],
            'type_of_property':[type_of_property]}

data = pd.DataFrame.from_dict(building)
data = fill(data)



pipe = load(open('preprocessor.pkl', 'rb'))
model_xgb = xgb.XGBRegressor()
model_xgb.load_model("model.json")

left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Estimate price:')
if pressed:
    X = pipe.transform(data)
    preds = model_xgb.predict(X)
    right_column.write(str(int(np.exp(preds)))+' €')

