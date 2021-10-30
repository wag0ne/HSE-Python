# AdvancedPython
**Задание 1**
\
Реализовать API, которое умеет:
1. Обучать ML-модель с возможностью настройки
гиперпараметров. При этом гиперпараметры для разных
моделей могут быть разные. Минимальное количество классов
моделей доступных для обучения == 2.
2. Возвращать список доступных для обучения классов моделей
3. Возвращать предсказание конкретной модели (как следствие,
система должна уметь хранить несколько обученных моделей)
4. Обучать заново и удалять уже обученные модели

api_flask.py - скрипт, который создает Flask API с которым общаемся через request.get(..)
\
api_models.py - класс реализации работы с данными, определением типа мл-задачи, обучение, предсказание, подсчет метрики.
