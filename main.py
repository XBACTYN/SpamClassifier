
import nltk
#Запустить единожды
#nltk.download('punkt')
#nltk.download('stopwords')
import numpy as np
import copy
import pandas as pd
import re
import pymorphy2 as pymorphy2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

vectorizer = CountVectorizer(ngram_range=(2, 2))
classificator = MultinomialNB()

def tokenize(str):
    morph = pymorphy2.MorphAnalyzer()
    #Удаление знаков препинания
    str = re.sub(r'[^\w\s]', ' ', str)
    #Токенизация
    tokens = word_tokenize(str)
    #Фильтр стоп-слов
    tokens = [i.lower() for i in tokens if (i not in stopwords.words('english'))]
    #Нормализация
    tokens = [morph.parse(tk)[0].normal_form for tk in tokens]
    #Склеиваем в строку, возвращаем
    return " ".join(tokens)


def to_binary(val):
    if val == "spam":
        return 1
    else:
        return 0


def calc_IDF(matrix):
    normal_matrix = copy.copy(matrix)
    N = len(normal_matrix)
    for i in range(0, N):
        normal_matrix[i] = normal_matrix[i].astype(bool)
    DF = np.sum(normal_matrix, axis=0) + 1e-5
    IDF = np.log(N / DF)
    return IDF


def calc_confusion_matrix(fact, predict):
    confusion_matrix = np.array([[0, 0], [0, 0]])
    for i in range(0, len(fact)):
        confusion_matrix[predict[i], fact[i]] += 1
    confusion_matrix = {"TP": confusion_matrix[0, 0], "FP": confusion_matrix[0, 1],
                        "FN": confusion_matrix[1, 0], "TN": confusion_matrix[1, 1]}
    return confusion_matrix


def report(confusion):
    #Precison - правильно классифицированные ham(1)
    precision = confusion['TP']/(confusion['TP']+confusion['FP'])
    #Доля найденных ham
    recall = confusion['TP']/(confusion['TP']+confusion['FN'])
    return precision,recall

def F_measure(precision,recall,beta):
    F = (1+beta**2)*(precision*recall)/((precision*beta**2)+recall)
    return F

if __name__ == '__main__':
    path = "src/spam.csv"
    #Извлекаем таблицу
    data = pd.read_csv(path, sep=",", usecols=[0, 1], dtype={"v1": str, "v2": str}, encoding='latin-1')
    print(data)
    #Преобразуем в таблицу из pandas
    df = pd.DataFrame({"v1": data["v1"], "v2": data["v2"]})
    #Поэлементно работаем со столбцами.
    df["v1"]= df["v1"].apply(to_binary)
    df["v2"]= df["v2"].apply(tokenize)
    #print(df)

    X = df["v2"].values
    y = df["v1"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # матрица отображающая биграммы слов
    # столбцы это возможные пары слов, строки это message
    TF_matrix = vectorizer.fit_transform(X_train).toarray()
    print(TF_matrix)
    IDF_vector = calc_IDF(TF_matrix)
    TF_IDF = TF_matrix * IDF_vector
    classificator.fit(X=TF_IDF, y=y_train)

    # для просмотра TF_IDF
    df_out = pd.DataFrame(data=TF_IDF, columns=vectorizer.get_feature_names_out())  # 3900 х ...
    print(df_out)

    test_TF_matrix = vectorizer.transform(X_test).toarray()
    test_IDF_vector = calc_IDF(test_TF_matrix)
    test_TF_IDF = test_TF_matrix * test_IDF_vector
    predicts = classificator.predict(test_TF_IDF)

    confusion = calc_confusion_matrix(y_test, predicts)
    print(f"confusion matrix:\nTP: {confusion['TP']}\t|\tFP: {confusion['FP']}\n"
          f"FN: {confusion['FN']}\t\t|\tTN: {confusion['TN']}\n")

    precision, recall = report(confusion)
    beta =1 #Вес точности
    F = F_measure(precision,recall,beta)
    print("Precision -  Процент истинных ham в проклассифицированных ham")
    print("Recall    -  Процент  правильно проклассифицированных ham из всех ham")
    print("F-мера    -  Среднее гармоническое (мера точности)")
    print(f"Precision:\t{precision*100}%\nRecall:\t{recall*100}%\nF-мера:\t{F*100}%")


