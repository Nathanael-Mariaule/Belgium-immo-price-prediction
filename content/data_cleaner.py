import pandas as pd



def eliza_cleaning(datas):
    datas.drop(columns=['Unnamed: 0'], inplace=True)
    old_col=(datas.columns)
    new_col=[]
    # rename column in a pythonic way
    for item in old_col:
        item =item.lower()
        item= item.replace(' ','_')
        new_col.append(item)

    datas.columns = new_col
    datas = datas[~datas['price'].isnull()]
    columns_restricted = ['locality', 'type_of_property', 'subtype_of_property',
           'price', 'type_of_sale', 'number_of_rooms', 'area',
           'fully_equipped_kitchen', 'furnished', 'open_fire', 'terrace',
           'terrace_area', 'garden', 'garden_area', 'surface_of_the_land',
           'surface_area_of_the_plot_of_land', 'number_of_facades',
           'swimming_pool', 'state_of_the_building'] # columns name without url and source
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
    datas = datas[~datas['price'].isnull()]
    datas.drop(columns =['source', 'url'], inplace=True)
    prices = datas.pop('price')
    datas.insert(0, 'price', prices)
    datas = datas[((datas['price']<3*10**7) & (datas['price']>2500)) | (datas.price.isnull())]
    datas = datas[(datas['number_of_rooms']<100) | (datas.number_of_rooms.isnull())]
    datas = datas[(datas['area']<755000) | (datas.area.isnull())]
    datas = datas[(datas['terrace_area']<3000) | (datas.terrace_area.isnull())]
    datas = datas[(datas['garden_area']<100000) | (datas.garden_area.isnull())]
    datas = datas[(datas['surface_of_the_land']<100000) | (datas.surface_of_the_land.isnull())]
    datas = datas[(datas['number_of_facades']<5) | (datas.number_of_facades.isnull())]
    datas.drop(columns=['surface_area_of_the_plot_of_land', 'type_of_sale', 'subtype_of_property'], inplace=True)
    datas.drop(columns=['subtype_of_property'], inplace=True)
    datas['garden_area'] = datas['garden_area'].where(datas['garden'] == 1, other=datas['garden'])
    datas['terrace_area'] = datas['terrace_area'].where(datas['terrace'] == 1, other=datas['terrace'])
    return datas



