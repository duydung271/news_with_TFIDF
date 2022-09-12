import json
import re

def get_data(path = 'Crawler/khoa-hoc.json'):
    f = open(path,encoding = 'utf-8-sig')
    data = json.load(f)
    f.close()
    return data

def get_stopwords(path='Classifier/stopwords.txt'):
    f = open(path)
    stop_words = [stop_word for stop_word in f.read().split('\n')]
    f.close()
    return stop_words

def clean_text(text, stop_words, word_len_limit = 20):

    # Clean (convert to lowercase and remove punctuations and characters and then strip)
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove number
    text = re.sub(r'\d', ' ', text)
    # Remove duplicate space
    text = re.sub(r' +', ' ', text)

    text = [word.lower() for word in text.split() if ((word.lower() not in stop_words) and (len(word) < word_len_limit))]
    text = " ".join(text) #removing stopwords

    return text

def choice_content(item):
    return item['title'] +' '+ item['description']+ ' ' + item['content']