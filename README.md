# Двухуровневая рекомендательная система

### Данные
Работаем c данными [MovieLens](https://grouplens.org/datasets/movielens/) 
делаем разбиение для двух уровней - можно посмотреть в `src/preprocess.py`

В результате разбиения получаем: 

    stage1_train - обучаем модели первого уровня
    stage2_predict - генерируем кандидатов с помощью моделей первого уровня и подаем их в катбуст
    stage2_train - данные для обучения катбуста на 2ом уровне
    stage2_holdout - для подсчета итоговой метрики всего пайплайна и оптимизации под нее

    final_training - тестовые данные для обучения моделей 1го уровня
    final_testset - генерируем кандидатов и переранжируем с помощью обученного катбуста
    final_holdout - данные для подсчета финальных метрик

### Baseline модели
- **SASRec**

Реализация sequentional модели [из статьи](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf)
Обучение модели смотреть в ноутбуке `notebooks/SASRec.ipynb`

### Модели 1го уровня
- **EASEr** ([статья](https://arxiv.org/pdf/1905.03375v1.pdf))
- **TensorModel**  ([статья](https://arxiv.org/pdf/1607.04228.pdf))

### Модели 2го уровня
- **CatBoostClassifier**


Запуск проекта:
```
poetry install
poetry run python3 main.py hydra.job.chdir=False
```