import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Carregar o dataset contendo ironias
data = pd.read_csv('irony_dataset.csv')

# Carregar o banco de dados de palavras positivas e negativas
sentiment_data = pd.read_csv('sentiment_words.csv')

# Pré-processamento dos dados, se necessário
# ...

# Separar features e labels
X = data['tweet_text']
y = data['irony_label']

# Vetorizar os tweets utilizando o TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Dividir o dataset em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Treinar o modelo SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Carregar as palavras positivas e negativas
positive_words = sentiment_data['positive_words'].tolist()
negative_words = sentiment_data['negative_words'].tolist()

# Classificar novos tweets fornecidos pelo usuário
subject = input("Digite o assunto a ser analisado: ")
tweets = [input("Digite um tweet: ")
          for _ in range(10)]  # Exemplo com 10 tweets

# Vetorizar e classificar os novos tweets
X_new = vectorizer.transform(tweets)
predictions = svm.predict(X_new)

# Contar a quantidade de tweets positivos e negativos
positive_tweets = 0
negative_tweets = 0

for i in range(len(predictions)):
    if predictions[i] == 'positivo':
        positive_tweets += 1
    elif predictions[i] == 'negativo':
        negative_tweets += 1
    else:  # Tratamento para tweets irônicos
        tweet_words = tweets[i].lower().split()
        positive_count = sum(word in positive_words for word in tweet_words)
        negative_count = sum(word in negative_words for word in tweet_words)

        if positive_count > negative_count:
            negative_tweets += 1
        else:
            positive_tweets += 1

# Imprimir resultados
print(f"Total de tweets positivos sobre '{subject}': {positive_tweets}")
print(f"Total de tweets negativos sobre '{subject}': {negative_tweets}")
