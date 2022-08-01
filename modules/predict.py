import os
import sys
import json
import dill as dill
import pandas as pd

path = '/opt/airflow/'
# Добавим путь к коду проекта в переменную окружения, чтобы он был доступен python-процессу
os.environ['PROJECT_PATH'] = path
# Добавим путь к коду проекта в $PATH, чтобы импортировать функции
sys.path.insert(0, path)


# Загрузка обученной модели, airflow.


def predict():

    with open(f'{path}/data/models/cars_pipe.pkl', 'rb') as file:
        model = dill.load(file)

    predict_df = []

    # Делает предсказания для всех объектов в папке data/test.
    for filename in os.listdir('f{path}/data/test'):
        with open(f'{path}/data/test/' + filename, 'r') as f:
            date = json.loads(f.read())
            predict_df.append(date)

        df = pd.DataFrame(predict_df)
        y = model.predict(df)

        df_result = pd.DataFrame(zip(df.id, y))
        df_result.columns = [['id', 'predict']]

        df_result.to_csv(f'{path}/data/predictions/pre_cars.csv, index=False')  # выводит результат в csv-файл.


if __name__ == '__main__':
    predict()
