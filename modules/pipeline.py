import logging
import os
import sys
from datetime import datetime

import dill
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


path = '/opt/airflow/'
# Добавим путь к коду проекта в переменную окружения, чтобы он был доступен python-процессу
os.environ['PROJECT_PATH'] = path
# Добавим путь к коду проекта в $PATH, чтобы импортировать функции
sys.path.insert(0, path)

#  Очистка днных.
def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]
    return df.drop(columns_to_drop, axis=1)

#  Очистка выбросов.
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        bounds = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        return bounds

    df = df.copy()
    boundaries = calculate_outliers(df['year'])
    df.loc[df['year'] < boundaries[0], 'year'] = round(boundaries[0])
    df.loc[df['year'] > boundaries[1], 'year'] = round(boundaries[1])
    return df

# Генерация фичей.
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    def short_model(x):
        if not pd.isna(x):
            return x.lower().split(' ')[0]
        else:
            return x

    df = df.copy()
    df.loc[:, 'short_model'] = df['model'].apply(short_model)
    df.loc[:, 'age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
    return df

#  Запускаем конвейер. 
def pipeline():
    #  Для airflow.
    df = pd.read_csv(f'{path}/data/train/homework.csv')
    # Локально.
    #df = pd.read_csv('C:.../airflow_hw/data/train/homework.csv')

    X = df.drop('price_category', axis=1)
    y = df['price_category']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])  # Выделяем числовые фичи.
    categorical_features = make_column_selector(dtype_include=object)  # Выделяем категориальные фичи. 

    numerical_transformer = Pipeline(steps=[  #  Привеодим числовые фичи к "единому масштабу".
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[  #  "Оцифровываем" категориальные переменные. 
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[  #  Объединяем полученные категориальные и числовые фичи.
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[  # Запускаем предобработку данных. 
        ('filter', FunctionTransformer(filter_data)),
        ('outlier_remover', FunctionTransformer(remove_outliers)),
        ('feature_creator', FunctionTransformer(create_features)),
        ('column_transformer', column_transformer)
    ])

    models = [  #  Список применяемых моделей. 
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    ]

    best_score = .0
    best_pipe = None
    for model in models:  #  Отработка моделей и выявление наиболее точной. 

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        logging.info(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    logging.info(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')

    best_pipe.fit(X, y)
    #  Для airflow.
    model_filename = f'{path}/data/models/cars_pipe.pkl'
    #  Локально.
    #model_filename = f'C:.../airflow_hw/data/models/cars_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.pkl'

    with open(model_filename, 'wb') as file:
        dill.dump(best_pipe, file)

    logging.info(f'Model is saved as {model_filename}')


if __name__ == '__main__':
    pipeline()
