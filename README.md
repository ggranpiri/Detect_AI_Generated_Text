# Detect_AI_Generated_Text
Нейросеть, которая предсказывает вероятность того, что текст сгенерирован искусственным интеллектом

Данный проект выполнен в рамках участия в соревновании Kaggle 

## Описание проекта

### Архитектура
Модель была построена с использованием библиотеки TensorFlow и Keras. Архитектура модели включает в себя три полносвязных (Dense) слоя:

- **Первый слой:** 128 нейронов, функция активации ReLU.
- **Второй слой:** 64 нейрона, функция активации ReLU.
- **Выходной слой:** 1 нейрон, функция активации Sigmoid.

Для предотвращения переобучения используется два слоя Dropout с коэффициентом 0.3.

### Данные
В проекте использованы два датасета:
1. **train_essays.csv** - оригинальные данные с соревнование Kaggle
2. **train_v2_drcat_02.csv** - дополнительные данные, так как соревновательный датасет очень маленький и содержит всего 3 эссе, сгенерированных нейросетью

### Предобработка данных
Выполнено удаление дубликатов и пустых строк.
Для векторизации текстов использован TF-IDF векторизатор с максимальным количеством признаков, равным 5000. Данные были разделены на обучающую и тестовую выборки в соотношении 80/20. Целевые метки были закодированы в бинарный формат (0 и 1) с использованием LabelEncoder.

### Обучение модели
Модель была обучена на протяжении 10 эпох с использованием оптимизатора Adam и функцией потерь binary_crossentropy. Размер батча составил 32.

### Результаты
Модель достигла точности 99% на тестовых данных, 82% на соревновании Kaggle, что свидетельствует о хорошем уровне производительности для задачи классификации текстов по авторству.
Это пример того, что даже достаточно простая нейросеть может выполнять сложные задачи.

### Сохранение модели
Обученная модель и TF-IDF токенизатор были сохранены для дальнейшего использования в следующие файлы:

- **Модель:** `text_classification_model.h5`
- **Токенизатор:** `tfidf_tokenizer.pkl`

## Итоги
Проект демонстрирует возможность использования нейросетевых моделей для решения задачи классификации текстов по авторству с высокой точностью. Модель и методы предобработки могут быть адаптированы для других задач, связанных с анализом текстов.
