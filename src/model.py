from typing import Any
from sklearn.ensemble import GradientBoostingRegressor


class TaxiFareModel:
    """Класс работы с моделью GradientBoostingRegressor"""

    def __init__(self):
        self.model = GradientBoostingRegressor()

    def fit(self, x, y):
        """     Тренировка модели
                Args:
                    x: Матрица признаков.
                    y: Целевая значение.
        """
        self.model.fit(x, y)

    def predict(self, x) -> Any:
        """     Расчте прогноз модели
                        Args:
                            x: Матрица признаков.
                        Returns:
                            Any прогноз модели
        """
        return self.model.predict(x)

    def score(self, x, y) -> float:
        """     R^2 Скоринг модели на тестовой выборке
                        Args:
                            x: Матрица признаков.
                            y: Целевая переменная
                        Returns:
                        float скоринг модели
        """
        return self.model.score(x, y)
