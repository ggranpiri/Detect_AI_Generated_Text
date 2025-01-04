import pickle
import re
from tensorflow.keras.models import load_model


def clean_text(text):
    """
    Очищает текст от лишних символов, приводит к нижнему регистру.
    """
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s]", "", text)  # Удаляем все, кроме букв, цифр и пробелов
    text = text.strip().lower()  # Убираем пробелы по краям и приводим к нижнему регистру
    return text


def loader_model():
    """
    Загружает модель и токенизатор из файлов.
    """
    # Загружаем модель
    model = load_model("model/text_classification_model.h5")

    # Загружаем токенизатор
    with open("model/tfidf_tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer


def predict_text(text, model, tokenizer):
    """
    Выполняет предсказание вероятности, что текст сгенерирован ИИ.

    :param text: Строка текста для анализа
    :param model: Загруженная модель
    :param tokenizer: Загруженный токенизатор TF-IDF
    :return: Вероятность принадлежности текста ИИ в процентах
    """
    # Предобработка текста
    text = clean_text(text)

    # Векторизация текста
    text_vector = tokenizer.transform([text]).toarray()

    # Предсказание вероятности
    prediction = model.predict(text_vector)[0][0]
    return prediction * 100
