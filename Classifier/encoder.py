from sklearn.feature_extraction.text import TfidfVectorizer
from tools.preprocess import *
import pickle

data = get_data()
stopwords = get_stopwords()

clean_data = [clean_text(choice_content(item), stopwords) for item in data ]

vectorizer = TfidfVectorizer(
    min_df = 4,
    max_df = 0.95,
    max_features = 8000,
)

vectorizer.fit(clean_data)

file = open('Classifier/vectorizer.cache', 'wb')
pickle.dump(vectorizer, file)
file.close()

print('save vectorizer success !')




