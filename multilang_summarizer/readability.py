from textstat.textstat import textstat
from multilang_summarizer.spanish_syllabicator import Silabicador

def flesch_kincaid(tokenized_sentences):
    total_syllables = 0
    total_words = 0
    total_sentences = len(tokenized_sentences)
    for tokenized_sent in tokenized_sentences:
        for token in tokenized_sent:
            total_words += 1
            total_syllables += textstat.syllable_count(token)

    score = 206.835
    score -= 1.015 * (total_words / total_sentences)
    score -= 84.6 * (total_syllables / total_words)
    return score / 100

def szigriszt_pazos(tokenized_sentences):
    syl = Silabicador()
    total_syllables = 0
    total_words = 0
    total_sentences = len(tokenized_sentences)
    for tokenized_sent in tokenized_sentences:
        for token in tokenized_sent:
            total_words += 1
            total_syllables += len(syl(token)[0])
    score = 206.835
    score -= 62.3 * (total_syllables / total_words)
    score -= total_words / total_sentences
    return score / 100

def invert(method, tokenized_sentences):
    return 1 - method(tokenized_sentences)
