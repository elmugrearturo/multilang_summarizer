import pathlib
import os

from multilang_summarizer.lemmatizer import Lemmatizer, language_index, nltk_stopwords
from multilang_summarizer.summarizer import summarizer

available_languages = []
for key in language_index:
    if key in nltk_stopwords.keys():
        available_languages.append(key)
