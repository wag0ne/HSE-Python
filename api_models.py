import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import os
from flask import abort, Response

from collections import defaultdict

from sklearn.linear_model import LogisticRegression, Ridge
from catboost import CatBoostClassifier, CatBoostRegressor

from sklearn.metrics import roc_auc_score, r2_score, f1_score

from typing import List, Tuple, Dict, Union, Any

metrics_dict = {'regression': r2_score,
                'binary': roc_auc_score,
                'multiclass': f1_score}

models_dict = {'regression': [Ridge, CatBoostRegressor],
               'binary': [LogisticRegression, CatBoostClassifier],
               'multiclass': [CatBoostClassifier]}


class ML_models:
    def __init__(self):
        self.models = []
        self.fitted_models = []
        self.counter = 0
        self.task_type = None
        self.available_models = defaultdict()

    def _get_task_type(self, target: dict, cutoff: int = 10) -> None:
        """
        Автоматически определяет тип мл-задачи

        target: наш массив с целевой переменной
        cutoff: порог отсечения для определения задачи регрессии
        """
        if not isinstance(target, DataFrame):
            target = pd.DataFrame(target)

        if target.nunique()[0] == 2:
            self.task_type = 'binary'
        elif target.nunique()[0] > cutoff:
            self.task_type = 'regression'
        else:
            self.task_type = 'multiclass'

    def get_available_model(self, target: dict) -> str:
        """
        Определяет тип задачи и выводит доступные модели
        """
        self._get_task_type(target)
        print(f"Your task type -> {self.task_type}")
        self.available_models[self.task_type] = {md.__name__: md for md in models_dict[self.task_type]}
        to_print = [md.__name__ for md in models_dict[self.task_type]]
        return f'  Available models for current task: {to_print}'

    def create_model(self, model_name: str = '', **kwargs):
        """
        model_name: название модели, которое выбирает пользователь
        dataset_name: наименование датасета

        return: {
                'model_id' - id модели,
                'model_name' - название модели
                'task_type' -  тип задачи (определяется автоматически),
                'model' - обученная в дальнейшем модель,
                'scores' - метрика качества,
        }
        """
        self.counter += 1
        ml_dic = {'model_id': self.counter,
                  'model_name': None,
                  'task_type': self.task_type,
                  'model': 'Not fitted',
                  'scores': {}}

        fitted = {'model_id': self.counter,
                  'model': 'Not fitted'}

        if model_name in self.available_models[self.task_type]:
            ml_dic['model_name'] = model_name
        else:
            self.counter -= 1
            abort(Response('''Wrong model name {}{}'''.format(model_name, self.available_models[self.task_type])))
        self.models.append(ml_dic)
        self.fitted_models.append(fitted)
        return ml_dic

    def get_model(self, model_id: int) -> Dict:
        """
        model_id: id модели
        """
        for model in self.models:
            if model['model_id'] == model_id:
                return model
        abort(Response('ml model {} doesnt exist'.format(model_id)))

    def _get_fitted_model(self, model_id: int) -> Dict:
        """
        model_id: id модели
        """
        for fit_model in self.fitted_models:
            if fit_model['model_id'] == model_id:
                return fit_model

    def update_model(self, model_dict: dict) -> None:
        """
        Метод обновляет содержимое словаря модели.
        """
        try:
            ml_model = self.get_model(model_dict['model_id'])
            ml_model.update(model_dict)
        except KeyError:
            abort(Response('Incorrect dictionary passed.'))
        except TypeError:
            abort(Response('Dictionary should be passed.'))

    def delete_model(self, model_id: int) -> None:
        """
        model_id: id модели, которую хотим удалить
        """
        model = self.get_model(model_id)
        fitted_model = self._get_fitted_model(model_id)
        self.fitted_models.remove(fitted_model)
        self.models.remove(model)

    @staticmethod
    def _transform_data(X, target) -> Tuple[DataFrame, Union[DataFrame, Any]]:
        """
        X: Обучающая выборка,
        target: Целевая переменная,
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(target, pd.DataFrame):
            target = pd.DataFrame(target)

        return X, target

    def fit(self, model_id, X, y, **kwargs) -> Dict:
        """
        model_id: id модели,
        X: Обучающая выборка,
        target: Целевая переменная,
        """
        X, y = self._transform_data(X, y)

        model_dic = self.get_model(model_id)
        fitted_model = self._get_fitted_model(model_id)

        if self.task_type == 'multiclass':
            params = {'random_state': 1488, 'loss_function': 'MultiClass'}
            algo = self.available_models[self.task_type][model_dic['model_name']](**params)
        else:
            params = {'random_state': 1488}
            algo = self.available_models[self.task_type][model_dic['model_name']](**params)

        try:
            algo.fit(X, y)
        except ValueError as error:
            abort(Response(f'{error}'))

        model_dic['model'] = 'Fitted'
        fitted_model['model'] = algo
        return model_dic

    def predict(self, model_id, X, to_json: bool = True, **kwargs) -> Union[DataFrame, Any]:
        """
        model_id: id модели,
        X: выборка для предсказания,
        to_json: завернуть ли предсказания в json формат
        """
        X = pd.DataFrame(X)
        _ = self.get_model(model_id)
        fitted_model = self._get_fitted_model(model_id)
        model = fitted_model['model']

        try:
            predict = model.predict(X)
        except AttributeError:
            abort(Response('Model not fitted yet, pleas use model.fit(..), before model.predict(..)'))

        if to_json:
            return pd.DataFrame(predict).to_json()
        return predict

    def predict_proba(self, model_id, X, to_json: bool = True, **kwargs) -> Union[DataFrame, Any]:
        """
        model_id: id модели,
        X: выборка для предсказания,
        to_json: завернуть ли предсказания в json формат
        """
        X = pd.DataFrame(X)
        fitted_model = self._get_fitted_model(model_id)
        model = fitted_model['model']
        try:
            if self.task_type == 'multiclass':
                try:
                    model_scores = model.predict_proba(X)
                except AttributeError:
                    abort(Response('Model not fitted yet, pleas use model.fit(..), before model.predict_proba(..)'))
            elif self.task_type == 'binary':
                try:
                    model_scores = model.predict_proba(X)[:, 1]
                except AttributeError:
                    abort(Response('Model not fitted yet, pleas use model.fit(..), before model.predict_proba(..)'))
        except AttributeError:
            abort(Response(f'Models with task_type {self.task_type} has no method predict_proba'))

        if to_json:
            return pd.DataFrame(model_scores).to_json()
        return model_scores

    def get_scores(self, model_id, X, y, **kwargs) -> Dict:
        """
        model_id: id модели,
        X: выборка для предсказания,
        y: истинные значения для посчета метрики качества
        """
        model_dic = self.get_model(model_id)

        X, y = self._transform_data(X, y)

        if X is None and y is None:
            abort(Response('For prediction needs X and y data!'))

        if self.task_type != 'regression':
            y_predicted = self.predict_proba(model_id, X, to_json=False)
        else:
            y_predicted = self.predict(model_id, X, to_json=False)

        metrics = metrics_dict[self.task_type](y, y_predicted)
        print(f"Models metrics {metrics_dict[self.task_type].__name__} = {metrics}")
        model_dic['scores'] = metrics
        return model_dic
