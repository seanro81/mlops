import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
        Загружаем данные из CSV-файла.

        Args:
            path: Путь к файлу с данными.

        Returns:
            DataFrame с загруженными данными.
    """
    return pd.read_csv(path)


def split_data(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
       Разделение выборки на Тестовую и Тренировочную

       Args:
            data: сэмпл с данными.

       Returns:
            DataFrame с загруженными данными
            DataFrame с загруженными данными
            DataFrame с загруженными данными
            DataFrame с загруженными данными
    """
    from sklearn.model_selection import train_test_split
    features = data.drop('fare_amount', axis=1)
    target = data['fare_amount']
    return train_test_split(features, target, test_size=0.2)

