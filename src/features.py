import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
           Добавим к выборке перобразование pickup_datetime в
           часы,день недели, а также почистим от key

           Args:
                df : сэмпл с данными.

           Returns:
                DataFrame с загруженными данными
    """
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    return df.drop(['key', 'pickup_datetime'], axis=1)
