import os
from collections import OrderedDict

language_index = {"ast" : "Asturian",
                  "bg" : "Bulgarian",
                  "ca" : "Catalan",
                  "cs" : "Czech",
                  "cy" : "Welsh",
                  "de" : "German",
                  "en" : "English",
                  "es" : "Spanish",
                  "et" : "Estonian",
                  "fa" : "Farsi",
                  "fr" : "French",
                  "ga" : "Irish",
                  "gd" : "Scottish Gaelic",
                  "gl" : "Galician",
                  "gv" : "Manx Gaelic",
                  "hu" : "Hungarian",
                  "it" : "Italian",
                  "pt" : "Portuguese",
                  "ro" : "Romanian",
                  "sk" : "Slovak",
                  "sl" : "Slovene",
                  "sv" : "Swedish",
                  "uk" : "Ukrainian"}

nltk_stopwords = {"ar" : 'arabic',
                  "az" : 'azerbaijani',
                  "da" : 'danish',
                  "nl" : 'dutch',
                  "en" : 'english',
                  "fi" : 'finnish',
                  "fr" : 'french',
                  "de" : 'german',
                  "el" : 'greek',
                  "hu" : 'hungarian',
                  "in" : 'indonesian',
                  "it" : 'italian',
                  "kk" : 'kazakh',
                  "ne" : 'nepali',
                  "no" : 'norwegian',
                  "pt" : 'portuguese',
                  "ro" : 'romanian',
                  "ru" : 'russian',
                  "es" : 'spanish',
                  "sv" : 'swedish',
                  "tr" : 'turkish'}

class Lemmatizer(object):

    def __init__(self, term_dictionary, language_code, language_name):
        self._term_dictionary = term_dictionary
        self._language_code = language_code
        self._language_name = language_name
        self._lemmas = set([lemma for lemma in term_dictionary.values()])

    def __getitem__(self, key):
        try:
            return self._term_dictionary[key.lower()]
        except:
            if key.lower() in self._lemmas:
                return key.lower()
            return key

    def __len__(self):
        return len(self._term_dictionary)

    def __str__(self):
        return "%s (%s): %d lemmas" % (self._language_name,
                                       self._language_code,
                                       len(self))
    def __repr__(self):
        return str(self)

    def __call__(self, key):
        return self[key]

def lemma_index(language_code, languages_path="./languages/"):
    '''
    ast - Asturian
    bg - Bulgarian
    ca - Catalan
    cs - Czech
    cy - Welsh
    de - German
    en - English
    es - Spanish
    et - Estonian
    fa - Farsi
    fr - French
    ga - Irish
    gd - Scottish Gaelic
    gl - Galician
    gv - Manx Gaelic
    hu - Hungarian
    it - Italian
    pt - Portuguese
    ro - Romanian
    sk - Slovak
    sl - Slovene
    sv - Swedish
    uk - Ukrainian
    '''
    if languages_path.endswith(os.sep):
        languages_path = languages_path[:-1]

    file_path = "%s/lemmatization-%s.txt" % (languages_path, language_code)

    index = OrderedDict({})
    with open(file_path, "r") as f:
        for line in f:
            try:
                value, key = line.replace("\ufeff", "").strip().split("\t")
            except:
                print(language_index[language_code])
                pass
            index[key] = value
    return Lemmatizer(index, language_code, language_index[language_code])
