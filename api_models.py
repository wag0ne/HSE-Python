import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import os
from flask import abort, Response

from collections import defaultdict

from sklearn.linear_model import LogisticRegression, Ridge
from catboost import CatBoostClassifier, CatBoostRegressor

from sklearn.metrics import roc_auc_score, r2_score, f1_score

from typing import Tuple, Dict, Union, Any

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

    def _get_task_type(self, data: dict, cutoff: int = 10) -> None:
        """
        Автоматически определяет тип мл-задачи

        data: обучающая выборка у которой целевая переменная называется -> "target"
        cutoff: порог отсечения для определения задачи регрессии
        """

        target = pd.DataFrame(data)[['target']]

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
        self.available_models[self.task_type] = {md.__name__: md for md in models_dict[self.task_type]}
        to_print = [md.__name__ for md in models_dict[self.task_type]]
        return f"Current task '{self.task_type}':\n    Available models: {to_print}"

    def create_model(self, model_name: str = '', **kwargs) -> Dict:
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
        ml_dic = {
            'model_id': self.counter,
            'model_name': None,
            'task_type': self.task_type,
            'model': 'Not fitted',
            'scores': {},
        }

        fitted = {
            'model_id': self.counter,
            'model': 'Not fitted',
        }

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

    def get_fitted_model(self, model_id: int) -> Dict:
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
        fitted_model = self.get_fitted_model(model_id)
        self.fitted_models.remove(fitted_model)
        self.models.remove(model)

    @staticmethod
    def _get_dataframe(data: dict) -> Tuple[DataFrame, Union[DataFrame, Any]]:
        """
        data: Обучающая выборка, вместе с таргетом
        """
        X = pd.DataFrame(data).drop(columns='target')
        target = pd.DataFrame(data)[['target']]

        return X, target

    def fit(self, model_id, data, **kwargs) -> Dict:
        """
        model_id: id модели,
        data: Обучающая выборка, с таргетом
        """
        X, y = self._get_dataframe(data)

        model_dict = self.get_model(model_id)
        fitted_model = self.get_fitted_model(model_id)

        if self.task_type == 'multiclass':
            params = {'random_state': 1488, 'loss_function': 'MultiClass'}
            algo = self.available_models[self.task_type][model_dict['model_name']](**params)
        else:
            params = {'random_state': 1488}
            algo = self.available_models[self.task_type][model_dict['model_name']](**params)

        algo.fit(X, y)

        model_dict['model'] = 'Fitted'
        fitted_model['model'] = algo
        return model_dict

    def predict(self, model_id, X, to_dict: bool = True, **kwargs) -> Union[DataFrame, Any]:
        """
        model_id: id модели,
        X: выборка для предсказания, без таргета
        to_json: завернуть ли предсказания в json формат
        """
        X = pd.DataFrame(X)
        _ = self.get_model(model_id)
        fitted_model = self.get_fitted_model(model_id)
        model = fitted_model['model']

        predict = model.predict(X)

        if to_dict:
            return pd.DataFrame(predict).to_dict()
        return predict

    def predict_proba(self, model_id, X, to_dict: bool = True, **kwargs) -> Union[DataFrame, Any]:
        """
        model_id: id модели,
        X: выборка для предсказания, без таргета
        to_json: завернуть ли предсказания в json формат
        """
        X = pd.DataFrame(X)
        fitted_model = self.get_fitted_model(model_id)
        model = fitted_model['model']
        try:
            if self.task_type == 'multiclass':
                model_scores = model.predict_proba(X)
            elif self.task_type == 'binary':
                model_scores = model.predict_proba(X)[:, 1]
        except AttributeError:
            abort(Response(f'Models with task_type {self.task_type} has no method predict_proba'))

        if to_dict:
            return pd.DataFrame(model_scores).to_dict()
        return model_scores

    def get_scores(self, model_id, data, **kwargs) -> Dict:
        """
        model_id: id модели,
        data: выборка для предсказания, c целевой переменной
        """
        if data is None:
            abort(Response('LOL! For compute metric needs data!'))

        model_dict = self.get_model(model_id)

        X, y = self._get_dataframe(data)

        if self.task_type != 'regression':
            y_predicted = self.predict_proba(model_id, X, to_dict=False)
        else:
            y_predicted = self.predict(model_id, X, to_dict=False)

        metrics = metrics_dict[self.task_type](y, y_predicted)
        model_dict['scores'] = {metrics_dict[self.task_type].__name__: metrics}
        return model_dict
