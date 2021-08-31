# Import packages
import nltk; nltk.download('wordnet'); nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
import re
# from config import *

nltk.download('averaged_perceptron_tagger')


class PreProcessing():


    def __init__(self):
        self.lower = lower
        self.replace_words = replace_words
        self.remove_punctuation = remove_punctuation
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords


    def preprocessor(self, input):
        """ Accept text and preprocess the text based on the passed flags"""        
        lemmatizer = WordNetLemmatizer()
        stoplist = set(stopwords.words('english'))
        stoplist.update(('cm', 'kg', 'mr', 'wa', 'nv', 'ore', 'da', 'pm', 'am', 'cx'))
        stoplist.remove('not')

        if self.lower:
            input = input.lower()
        if self.replace_words:
            cleaned_description = []
            for word in str(input).split():
                if word.lower() in appos.keys():
                    cleaned_description.append(appos[word.lower()])
                else:
                    cleaned_description.append(word)
            input = ' '.join(cleaned_description)

        if self.remove_punctuation:
            PUNCT_TO_REMOVE = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' #string.punctuation
            input = input.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

        if self.lemmatize:        
            input = " ".join([lemmatizer.lemmatize(word) for word in str(input).split()])

        if self.remove_stopwords:
            input = " ".join([word for word in str(input).split() if word not in stoplist])

        #Removing multiple spaces between words
        input = re.sub(' +', ' ', input)
        return input