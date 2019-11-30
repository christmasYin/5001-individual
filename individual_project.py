import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import time
import datetime
import pickle


def data_preprocess(data):
    # 删除bool的列
    data = data.drop(['id','is_free'], axis=1)
    # 前向填充空缺值
    data = data.fillna(method = 'pad')

    # 处理date-time数据
    data_prepared = data.drop(['purchase_date', 'release_date'],axis=1)
    data_date = data.loc[:, ['purchase_date', 'release_date']]

    # 增加时间差的feature
    data_date['purchase_date'] = pd.to_datetime(data_date['purchase_date'])
    data_date['release_date'] = pd.to_datetime(data_date['release_date'])
    data_date['date_diff'] = data_date['purchase_date']-data_date['release_date']
    data_date['date_diff_days'] = data_date['date_diff'].map(lambda x:x.days)
    data_date.loc[data_date['date_diff_days'] < 0, 'date_diff_days'] = 0


    def extract_date(df, column):
        df[column + '_year'] = df[column].apply(lambda x: x.year)
        df[column + '_month'] = df[column].apply(lambda x: x.month)

    extract_date(data_date, 'purchase_date')
    extract_date(data_date, 'release_date')

    data_date = data_date.drop(['purchase_date', 'release_date','date_diff'], axis=1)
    data_prepared = pd.concat([data_prepared, data_date], axis=1)

    return data_prepared

def train(data, label):
    rf_model = RandomForestRegressor()
    rf_model.fit(data, label)
    scores = cross_val_score(rf_model, data, label, scoring='neg_mean_squared_error', cv=10)
    scores_rmse = np.sqrt(-scores)
    scores_rmse_mean = scores_rmse.mean()
    print(scores_rmse_mean)
    with open('save/model.pickle', 'wb') as f:
        pickle.dump(rf_model, f)

def predict(data):
    with open('save/model.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    predictions = loaded_model.predict(data)
    result = pd.DataFrame(predictions, columns=['playtime_forever'])
    residual = result.loc[result['playtime_forever'] < 1].mean(axis = 0)
    result = result-residual
    result.loc[result['playtime_forever'] < 0, 'playtime_forever'] = 0
    result.to_csv('prediction.csv',index_label=['id'])

if __name__ == '__main__':

    train_data = pd.read_csv('/Users/apple/Desktop/BDT/5001/msbd5001-fall2019/train.csv', parse_dates=['purchase_date', 'release_date'])
    train_data.sort_values('total_positive_reviews', inplace=True, ascending=False)
    train_data['playtime_forever'][0:20] = train_data['playtime_forever'][0:20]*3
    test_data = pd.read_csv('/Users/apple/Desktop/BDT/5001/msbd5001-fall2019/test.csv', parse_dates=['purchase_date', 'release_date'])

    # 分离feature和label
    train_data_label = train_data['playtime_forever']
    train_data = train_data.drop(['playtime_forever'], axis=1)
    train_data = data_preprocess(train_data)
    # perform one-hot encoding
    genres_dummies = train_data['genres'].str.get_dummies(',')
    categories_dummies = train_data['categories'].str.get_dummies(",")
    tags_dummies = train_data['tags'].str.get_dummies(",")

    #给列重命名，避免列名重复
    genres_dummies.rename(columns=lambda x: x + '_g', inplace=True)
    categories_dummies.rename(columns=lambda x: x + '_c', inplace=True)
    tags_dummies.rename(columns=lambda x: x + '_t', inplace=True)

    # replace 'genres' by 'genres_dummies'
    train_data = train_data.drop(['genres','categories','tags'], axis=1)
    train_data = pd.concat([train_data, genres_dummies, categories_dummies, tags_dummies], axis=1)

    pca = PCA(n_components=20)
    pca.fit(train_data)
    train_data_reduced = pca.transform(train_data)

    # save the keys
    keys = genres_dummies.keys()
    keys = keys.append([categories_dummies.keys(), tags_dummies.keys()])

    train(train_data_reduced,train_data_label)


    """ for testing data """

    test_data = data_preprocess(test_data)

    genres_dummies = test_data['genres'].str.get_dummies(',')
    categories_dummies = test_data['categories'].str.get_dummies(",")
    tags_dummies = test_data['tags'].str.get_dummies(",")

    genres_dummies.rename(columns=lambda x: x + '_g', inplace=True)
    categories_dummies.rename(columns=lambda x: x + '_c', inplace=True)
    tags_dummies.rename(columns=lambda x: x + '_t', inplace=True)

    # replace 'genres' by 'genres_dummies'
    test_data = test_data.drop(['genres','categories','tags'], axis=1)
    test_data = pd.concat([test_data, genres_dummies, categories_dummies, tags_dummies], axis=1)

    # deal with the missing key(s)
    for k in keys:
        if test_data.get(k) is None:
            test_data[k] = 0

    # sort the keys
    test_data = test_data[train_data.keys()]

    test_data_reduced = pca.transform(test_data)
    predict(test_data_reduced)
