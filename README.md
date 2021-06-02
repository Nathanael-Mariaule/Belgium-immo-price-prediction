# challenge-regression
_Author: Nathanaël Mariaule_

## 1/ The project:
In the context of Becode's AI, we were asked to build a model of price predictin for housing in Belgium. The dataset was previously collected and can be found at _'https://raw.githubusercontent.com/JulienAlardot/challenge-collecting-data/main/Data/database.csv'_. The file model_training train a XGBoost model on the dataset. This was the best model we could obtain w.r.t. the R2 metric.

## 2/ The files:
### data_cleaner.py
A python routine for preliminary cleaning of the dataset
### model_training.py
Code for the training of the model
### models_comparison.ipynb
Jupyter notebook with performance comparisons of various model
### preprocessing_comparison.ipynb
Jupyter notebook with performance comparisons of various preprocessing techniques
### hyperparameters tuning
Jupyter notebook with the hyperparameters tuning of the xgboost model
### model_loading.ipynb
Example of use of our model
### preprocessor.pkl
pipeline object from sklearn. Trained on our dataset
### model.json
XGBRegressor object from xgboost. Our model trained on the dataset
### post_codes.csv: 
Used to get the postcodes and names of areas.
[source](https://public.opendatasoft.com/explore/dataset/liste-des-codes-postaux-belges-fr/table/?flg=fr)
### median.csv
Used to get the median price for housing in each Belgian commune
[source](https://trends.knack.be/economie/immo/hoeveel-kost-een-woning-in-uw-gemeente-bekijk-de-interactieve-kaart/game-normal-1636503.html?cookie_check=1622441769)

## 3/ Improvements:
Due to lack of time, the following steps remain to do
- test Feature engineering to improve the score of the model.
- deployment of the model
