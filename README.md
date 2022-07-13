# Двухуровневая рекомендательная система

### Данные
Работаем c данными [MovieLens](https://grouplens.org/datasets/movielens/) 
делаем разбиение на train, test, holdout для двух уровней - можно посмотреть в `notebooks/split_data.ipynb`

### Модели 1го уровня
- **SASRec**

Реализация sequentional модели [из статьи](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf)
Обучение модели смотреть в ноутбуке `notebooks/SASRec.ipynb`




Запуск проекта:
```
poetry update
poetry run python3 main.py hydra.job.chdir=False
```