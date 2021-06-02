import pandas as pd



def eliza_cleaning(datas: pd.DataFrame)->pd.DataFrame:
    """
    the function take for input the dataset https://raw.githubusercontent.com/JulienAlardot/challenge-collecting-data/main/Data/database.csv'
    it changes the column names and remove useless datas
    :param datas: pandas DataFrame that contains the dataset
    :return datas: pandas DataFrame that contains the dataset after cleaning
    """
    datas.drop(columns=['Unnamed: 0'], inplace=True)
    old_col=(datas.columns)
    new_col=[]
    # rename column in a pythonic way
    for item in old_col:
        item =item.lower()
        item= item.replace(' ','_')
        new_col.append(item)
    datas.columns = new_col
    #keep the usefull columns
    columns_restricted = ['locality', 'type_of_property', 'subtype_of_property',
           'price', 'type_of_sale', 'number_of_rooms', 'area',
           'fully_equipped_kitchen', 'furnished', 'open_fire', 'terrace',
           'terrace_area', 'garden', 'garden_area', 'surface_of_the_land',
           'surface_area_of_the_plot_of_land', 'number_of_facades',
           'swimming_pool', 'state_of_the_building'] # columns name without url and source
    #change columns type
    datas = datas[~datas[columns_restricted].duplicated()]
    datas = datas.astype({'locality':'Int64', 
            'price':'float64', 
            'number_of_rooms':'Int64', 
            'area':'float64', 
            'fully_equipped_kitchen':'Int64', 
            'furnished':'Int64',
            'open_fire':'Int64',
            'terrace':'Int64',
            'terrace_area':'float64',
            'garden':'Int64',
            'garden_area':'float64',
            'surface_of_the_land':'float64',
            'surface_area_of_the_plot_of_land':'float64',
            'number_of_facades':'Int64',
            'swimming_pool':'Int64',
            })
    # drop datas with nan price
    datas = datas[~datas['price'].isnull()]
    datas.drop(columns =['source', 'url'], inplace=True)
    #move price columns at first place
    prices = datas.pop('price')
    datas.insert(0, 'price', prices)
    #move type_of_property columns last place
    type_of_property = datas.pop('type_of_property')
    datas['type_of_property'] = type_of_property
    #drop useless columns
    datas.drop(columns=['surface_area_of_the_plot_of_land', 'type_of_sale', 'subtype_of_property'], inplace=True)
    return datas

def eliza_fillna(datas: pd.DataFrame)->pd.DataFrame:
    """
        the function take for input the dataset https://raw.githubusercontent.com/JulienAlardot/challenge-collecting-data/main/Data/database.csv'
        it fill some nan values in the dataset
        :param datas: pandas DataFrame that contains the dataset
        :return datas: pandas DataFrame that contains the dataset after filling some nan's value
    """
    # if garden=0, we set garden_area as 0 same for terrace
    datas['garden_area'] = datas['garden_area'].where(datas['garden'] == 1, other=datas['garden'])
    datas['terrace_area'] = datas['terrace_area'].where(datas['terrace'] == 1, other=datas['terrace'])
    # add new category for nan value in state_of_the_building and type_of_property
    datas['state_of_the_building'].fillna(value='unkown', inplace=True)
    datas['type_of_property'].fillna(value='other', inplace=True)
    zeros = datas.fillna(value=0)
    mask = (datas.type_of_property=='apartment')  & (datas.surface_of_the_land.isnull()) & (datas.garden_area==0)
    #an appartement with no garden has 0 surface_of_the_land
    datas['surface_of_the_land'] = zeros.where(mask, other=datas).surface_of_the_land
    datas['furnished'].fillna(value=0, inplace=True)
    return datas
