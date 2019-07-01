import pathlib

import os
from functools import partial
from sentence_splitter import SentenceSplitter
import nltk.data
from nltk.corpus import stopwords
import pyphen

import re
import regex
import string

from collections import OrderedDict

PARENT_DIR = pathlib.Path(__file__).parent

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

punkt_tokenizers = {"da" : 'danish.pickle',
                    "et" : 'estonian.pickle',
                    "de" : 'german.pickle',
                    "no" : 'norwegian.pickle',
                    "sl" : 'slovene.pickle',
                    "tr" : 'turkish.pickle',
                    "nl" : 'dutch.pickle',
                    "fi" : 'finnish.pickle',
                    "el" : 'greek.pickle',
                    "pl" : 'polish.pickle',
                    "es" : 'spanish.pickle',
                    "cs" : 'czech.pickle',
                    "en" : 'english.pickle',
                    "fr" : 'french.pickle',
                    "it" : 'italian.pickle',
                    "pt" : 'portuguese.pickle',
                    "sv" : 'swedish.pickle'}

splitter_sent_tok = {"ca" : 'Catalan (ca)',
                     "cs" : 'Czech (cs)',
                     "da" : 'Danish (da)',
                     "nl" : 'Dutch (nl)',
                     "en" : 'English (en)',
                     "fi" : 'Finnish (fi)',
                     "fr" : 'French (fr)',
                     "de" : 'German (de)',
                     "el" : 'Greek (el)',
                     "hu" : 'Hungarian (hu)',
                     "is" : 'Icelandic (is)',
                     "it" : 'Italian (it)',
                     "lv" : 'Latvian (lv)',
                     "lt" : 'Lithuanian (lt)',
                     "no" : 'Norwegian (Bokmål) (no)',
                     "pl" : 'Polish (pl)',
                     "pt" : 'Portuguese (pt)',
                     "ro" : 'Romanian (ro)',
                     "ru" : 'Russian (ru)',
                     "sk" : 'Slovak (sk)',
                     "sl" : 'Slovene (sl)',
                     "es" : 'Spanish (es)',
                     "sv" : 'Swedish (sv)',
                     "tr" : 'Turkish (tr)'}

pyphen_dicts = {"de" : "de_DE",
                "en" : "en",
                "es" : "es",
                "fr" : "fr",
                "hu" : "hu_HU",
                "it" : "it",
                "pt" : "pt_BR",
                "ro" : "ro",
                "sv" : "sv"}


class Lemmatizer(object):

    def __init__(self, term_dictionary, language_code, language_name):
        self._term_dictionary = term_dictionary
        self._language_code = language_code
        self._language_name = language_name
        if self._language_code in punkt_tokenizers:
            splitter = nltk.data.load(
                "tokenizers/punkt/%s" % punkt_tokenizers[self._language_code]
                )
            self.sent_split = splitter.tokenize
        elif self._language_code in splitter_sent_tok:
            splitter = SentenceSplitter(language=self._language_code)
            self.sent_split = splitter.split
        else:
            # If nothing works, use naive sentence splitter
            self.sent_split = partial(re.split,
                                r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        self._lemmas = set([lemma for lemma in term_dictionary.values()])
        if self._language_code in nltk_stopwords:
            self._stopwords = stopwords.words(
                                            nltk_stopwords[self._language_code]
                                            )
        else:
            print("No stopwords:", self._language_code)

        # Change punctuation for whitespace
        self.remove_punctuation = partial(regex.sub, '[\p{P}]+', ' ')

        self.pyphen_dic = pyphen.Pyphen(lang=pyphen_dicts[self._language_code])

    def syllabicator(self, term):
        hyphenated = self.pyphen_dic.inserted(term)
        return hyphenated.split("-")

    def tokenize(self, text):
        unclean_tokenized_text = re.sub(r' +', ' ', text).split(" ")
        tokenized_text = [token for token in unclean_tokenized_text if token !=
                         ""]
        return tokenized_text

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

    def lemmatize(self, paragraph, remove_stopwords=True):

        # Separate sentences
        paragraph = paragraph.strip()
        sentences = self.sent_split(paragraph)

        tok_sentences = []
        lem_sentences = []
        # Separate tokens
        for sentence in sentences:
            tokens = self.tokenize(self.remove_punctuation(sentence))
            # Lemmatize
            lem_tokens = [self[token] for token in tokens]
            # Remove stopwords
            if remove_stopwords:
                lem_tokens = [token for token in lem_tokens if token not in
                              self._stopwords]
            tok_sentences.append(tokens)
            lem_sentences.append(lem_tokens)

        return sentences, tok_sentences, lem_sentences

    @classmethod
    def for_language(klass, language_code, languages_path=PARENT_DIR / "languages/"):
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

        file_path = languages_path / ("lemmatization-%s.txt" % language_code)

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
