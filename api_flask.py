from flask import Flask, request, jsonify, abort, Response
from flask_restx import Api
from api_models import ML_models

app = Flask(__name__)
api = Api(app)

my_models = ML_models()


@app.route("/api/get_possible_model", methods=['GET', 'PUT'])
def get_possible_model():
    """
    Выводит тип задачи и возможные модели для обучения
    target: для идентификации типа задачи
    """
    return my_models.get_available_model(request.json)


@app.route("/api/create_model", methods=['POST'])
def create_model():
    """
    Создает модель. В качестве параметров надо передать:
    model_name,
    """
    try:
        request.json['model_name']
    except KeyError:
        abort(Response("Please, check correct param 'model_name' "))
    my_models.create_model(**request.json)
    return 'Success'


@app.route("/api/get_model", methods=['GET'])
def get_all_models():
    """
    Выводит все модели
    """
    return jsonify(my_models.models)


@app.route("/api/get_model/<int:model_id>", methods=['GET'])
def get_model(model_id):
    """
    Выводит модель с указанным id
    """
    return my_models.get_model(model_id)


@app.route("/api/update_model", methods=['PUT'])
def update_model():
    """
    'model_name' - наименование модели
    """
    my_models.update_model(request.json)
    return 'Success'


@app.route("/api/delete_model/<int:model_id>", methods=['DELETE'])
def delete_model(model_id):
    """
    Удаление модели с указанным id
    """
    my_models.delete_model(model_id)
    return 'Success'


@app.route("/api/fit/<int:model_id>", methods=['PUT'])
def fit(model_id):
    """
    Обучение модели
    """
    my_models.fit(model_id, **request.json)
    return 'Success'


@app.route("/api/predict/<int:model_id>", methods=['GET', 'PUT'])
def predict(model_id):
    """
    Предсказания модели
    """
    preds = my_models.predict(model_id, **request.json)
    return preds


@app.route("/api/predict_proba/<int:model_id>", methods=['GET', 'PUT'])
def predict_proba(model_id):
    """
    Предсказания модели
    """
    model_scores = my_models.predict_proba(model_id, **request.json)
    return model_scores


@app.route("/api/get_scores/<int:model_id>", methods=['GET', 'PUT'])
def get_scores(model_id):
    """
    Возвращаются посчитанные метрики качества
    """
    scores = my_models.get_scores(model_id, **request.json)
    return scores


if __name__ == '__main__':
    app.run()
