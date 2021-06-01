import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, ElasticNetCV, LassoCV, RidgeCV, Lasso, SGDRegressor
from data_cleaner import eliza_cleaning, eliza_fillna
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, Normalizer, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import explained_variance_score as evs # evaluation metric
from sklearn.metrics import r2_score as r2 # evaluation metric
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from pickle import dump



raw_datas = pd.read_csv('https://raw.githubusercontent.com/JulienAlardot/challenge-collecting-data/main/Data/database.csv')
datas = eliza_cleaning(raw_datas)
datas = eliza_fillna(datas)

median = pd.read_csv('median.csv')
post = pd.read_csv('post_codes.csv', sep=';')
median['Gemeente'] = median['Gemeente'].str.lower()
post['Commune Principale'] = post['Commune principale'].str.lower()
median_with_post = median.merge(post[['Code postal', 'Commune Principale']], how='left', left_on='Gemeente', right_on='Commune Principale')
median_with_post = median_with_post.groupby('Gemeente').median()
median_with_post['Mediaanprijs 2020'].fillna(median_with_post['Mediaanprijs 2019'], inplace=True)
median_with_post['Mediaanprijs 2020'].fillna(median_with_post['Mediaanprijs 2018'], inplace=True)
median_with_post.sort_values(by='Code postal', inplace=True)
median_with_post.fillna(method='bfill', inplace=True)
median_with_post.reset_index(inplace=True)
median = median.merge(median_with_post[['Gemeente', 'Mediaanprijs 2020']], on='Gemeente')
median_with_post = median.merge(post[['Code postal', 'Commune Principale']], how='left', left_on='Gemeente', right_on='Commune Principale')
median_prices = median_with_post[['Code postal', 'Mediaanprijs 2020_y']]
median_prices.columns = ['postal_code', 'median_price']
median_prices = median_prices.groupby('postal_code').mean()
median_prices.reset_index(inplace=True)
median_prices['postal_code'] = median_prices['postal_code'].astype('int64')
datas = datas.merge(median_prices, how='left', left_on='locality', right_on='postal_code')
datas.drop('postal_code', inplace=True, axis=1)
datas.sort_values(by='locality', ascending=False, inplace=True)
datas['median_price'].fillna(method='ffill', inplace=True)
datas.sort_index(inplace=True)
col = datas.columns
col = [col[0]]+[col[-1]]+list(col[1:-1])
datas = datas[col]

y = datas.pop('price')
X = datas

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


fillna = ColumnTransformer(
        [('imp_col1', SimpleImputer(strategy='mean'), ['area', 'terrace_area', 'garden_area', 
                                                      'surface_of_the_land']),
         ('imp_col2', SimpleImputer(strategy='median'), ['number_of_rooms', 'number_of_facades']),
        ],remainder='passthrough')

enc = ColumnTransformer(
        [
         ('enc', OneHotEncoder(sparse = False, drop ='first'), [-4, -3,-2,-1]),
        ], remainder='passthrough')


pipe = make_pipeline(fillna, enc, MinMaxScaler(), StandardScaler())
pipe.fit(X)
X_train = pipe.transform(X_train)
X_val = pipe.transform(X_val)



my_XGB_model = XGBRegressor(n_estimators=3000, max_depth=11, min_child_weight=11,
                           learning_rate=0.01, subsample=1, colsample_bytree=1,
                           eval_metric='mae')
my_XGB_model.fit(X_train, np.log(y_train), early_stopping_rounds=10, 
             eval_set=[(X_val, np.log(y_val))], verbose=False)

X=pipe.transform(X)  
XGB_predictions = my_XGB_model.predict(X)
XGB_predictions = np.exp(XGB_predictions)
XGB_mult_mae = mean_absolute_error(XGB_predictions, y)
print("Validation MAE for multi-pass XGBoost Model : " + str(XGB_mult_mae))
              
  
preds = my_XGB_model.predict(X)
rmse = np.sqrt(mean_squared_error( y, np.exp(preds)))
print("RMSE: %f" % (rmse))
print("R2 score: ", my_XGB_model.score(X,  np.log(y)))

dump(pipe, open('preprocessor.pkl', 'wb'))
my_XGB_model.save_model("model.json")
