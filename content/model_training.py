import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, ElasticNetCV, LassoCV, RidgeCV
from data_cleaner import eliza_cleaning, eliza_fillna
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, Normalizer, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import mean_squared_error

raw_datas = pd.read_csv('https://raw.githubusercontent.com/JulienAlardot/challenge-collecting-data/main/Data/database.csv')
datas = eliza_cleaning(raw_datas)
datas = eliza_fillna(datas)
#datas.drop(columns=['locality'], inplace=True)
house = datas[datas['type_of_property']=='house'].copy()
appart = datas[datas['type_of_property']=='apartment'].copy()
y = datas.pop('price')
X = datas
house_y = house.pop('price')
house_x = house
appart_y = appart.pop('price')
appart_x = appart

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


fillna = ColumnTransformer(
        [ ('imp', KNNImputer(n_neighbors=2, weights="uniform"), list(range(1,12)))],
         remainder='passthrough')
fillna = ColumnTransformer(
        [('imp_col1', SimpleImputer(strategy='mean'), ['area', 'terrace_area', 'garden_area', 
                                                      'surface_of_the_land']),
         ('imp_col2', SimpleImputer(strategy='median'), ['number_of_rooms', 'number_of_facades']),
        ],remainder='passthrough')
enc = ColumnTransformer(
        [
         ('enc', OneHotEncoder(sparse = False, drop ='first'), [-4, -3,-2,-1]),
         #('enc2', OneHotEncoder(sparse = False, handle_unknown='ignore'), [-7])
        ], remainder='passthrough')


model = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=20, min_samples_leaf=10, 
                               n_jobs=-1, warm_start=True)
#model = ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio)

#pipe = make_pipeline(fillna, enc, Normalizer())
#pipe.fit(X_train)


#xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
#                max_depth = 5, alpha = 10, n_estimators = 10)
#xg_reg.fit(pipe.transform(X_train),y_train)
#preds = xg_reg.predict(pipe.transform(X_val))
#rmse = np.sqrt(mean_squared_error(y_val, preds))
#print("RMSE: %f" % (rmse))


alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
model = ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio)
model = LinearRegression()

pipe = make_pipeline(fillna, enc, Normalizer(), PolynomialFeatures(2), model )
pipe.fit(X_train, y_train)
preds = pipe.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
print("RMSE: %f" % (rmse))
print(pipe.score(X_val, y_val))
