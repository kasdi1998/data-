# importer les bibliothèques nécessaires
import numpy as np 
import pandas as pd 
import re
import nltk 
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

# charger les données
covid_tweets = pd.read_csv('vaccination_all_tweets.csv')

# extraire les fonctionnalités et les étiquettes
features = covid_tweets['text'].values
labels = covid_tweets['sentiment'].values

# prétraitement des fonctionnalités
processed_features = []

for sentence in features:
    # enlever tous les caractères spéciaux sauf les apostrophes
    processed_feature = re.sub(r'\W+', ' ', sentence)
    
    # remplacer plusieurs espaces par un seul espace
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    
    # convertir en minuscules
    processed_feature = processed_feature.lower()

    # séparer les mots négatifs en ajoutant "not_" ou "n't_" devant les mots suivants
    processed_feature = re.sub(r"(not|n't)\s+(\w+)", r"\1_\2", processed_feature)

    processed_features.append(processed_feature)

# Télécharger les stopwords et les punkt tokenizer de NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Tokeniser les phrases en mots
from nltk.tokenize import word_tokenize
processed_features_tokenized = []
for sentence in processed_features:
    # Tokenisation des phrases en mots
    words = word_tokenize(sentence)
    # Ajout des mots à la liste de fonctionnalités prétraitées
    processed_features_tokenized.append(' '.join(words))

# charger les stopwords en anglais
stop_words = stopwords.words('english')

# créer une matrice TF-IDF pour les données de fonctionnalités
vectorizer = TfidfVectorizer(max_features=2500, min_df=5, max_df=0.8, stop_words=stop_words)
processed_features_transformed = vectorizer.fit_transform(processed_features_tokenized).toarray()

# diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(processed_features_transformed, labels, test_size=0.2, random_state=0)

# créer un classificateur de Naive Bayes et l'entraîner avec les données d'entraînement
text_classifier = MultinomialNB()
text_classifier.fit(X_train, y_train)

# faire des prédictions sur les données de test
predictions = text_classifier.predict(X_test)

# afficher la matrice de confusion et le score de précision
print(confusion_matrix(y_test,predictions))
print('accuracy score',accuracy_score(y_test, predictions))

# fonction pour classer un nouveau texte
def classify_new(new_txt):
    #préparer le texte comme pour les mots du texte comme pour les données connues
    data = vectorizer.transform([new_txt]).toarray()
    return text_classifier.predict(data)

extra_txt = "the bus isn't good"
print(extra_txt, classify_new(extra_txt))
