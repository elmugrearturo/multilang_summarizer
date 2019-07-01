import pathlib

import copy
import os
import sys

import pickle

from collections import Counter

from functools import partial

from math import log

from multilang_summarizer.lcs import *

from multilang_summarizer.tfidf import calculate_tf, calculate_idf
from multilang_summarizer.entropy import syllable_metric_entropy

PARENT_DIR = pathlib.Path(__file__).parent

def get_likelihoods(lookup_table):
    likelihoods = {}
    terms = list(lookup_table.keys())
    values = list(lookup_table.values())
    total_seen = sum(values)
    for term in terms:
        likelihoods[term] = lookup_table[term] / total_seen
    return likelihoods

def overlapping_tokens(tokens1, tokens2):
    '''fake lcs from overlapping tokens'''
    all_tokens = []
    all_token_indexes = []
    modif_tokens1 = tokens1.copy()
    modif_tokens2 = tokens2.copy()
    for i, token in enumerate(modif_tokens1):
        if token in modif_tokens2:
            j = modif_tokens2.index(token)
            modif_tokens2[j] = " "
            all_tokens.append(token)
            all_token_indexes.append((i, j))
    return all_tokens, all_token_indexes

def probabilistic_tokens(tokens1, tokens2):
    '''fake lcs from overlapping tokens'''
    all_tokens = []
    all_token_indexes = []

    probabilities = []

    modif_tokens1 = tokens1.copy()
    modif_tokens2 = tokens2.copy()

    token_counter = Counter(modif_tokens1 + modif_tokens2)
    total = len(modif_tokens1) + len(modif_tokens2)

    for i, token in enumerate(modif_tokens1):
        if token in modif_tokens2:
            j = modif_tokens2.index(token)
            modif_tokens2[j] = " "
            all_tokens.append(token)
            all_token_indexes.append((i, j))
            probabilities.append(token_counter[token] / total)

    probabilities = list(enumerate(probabilities))
    probabilities.sort(key=lambda x: x[1], reverse=True)
    max_tokens = len(lcs(tokens1, tokens2))
    count = 0
    for i, prob in probabilities:
        if count < max_tokens:
            all_tokens[i] == None
            all_token_indexes[i] == None
            count += 1
    all_tokens = [t for t in all_tokens if t != None]
    all_token_indexes = [t for t in all_token_indexes if t != None]
    return all_tokens, all_token_indexes

def lcs_tokens(tokens1, tokens2):
    return lcs(tokens1, tokens2), lcs_indexes(tokens1, tokens2)

def create_index(aligned_sentences):
    index = {}
    start = 0
    sentence_number = 0
    for sent, tok_sent, lem_sent in aligned_sentences:
        for i in range(start, start + len(lem_sent)):
            index[i] = (sentence_number, sent, tok_sent, lem_sent)
        start += len(lem_sent)
        sentence_number += 1
    return index

class Document(object):

    def __init__(self, path, lemmatizer, remove_stopwords=True):
        self._lemmatizer = lemmatizer
        self._syllabicator = lemmatizer.syllabicator
        with open(path, "r") as f:
            self.raw_text = f.read()
        self.sentences = lemmatizer.lemmatize(self.raw_text,
                                              remove_stopwords=remove_stopwords)
        self.raw_sentences = self.sentences[0]
        self.tok_sentences = self.sentences[1]
        self.lem_sentences = self.sentences[2]

        self.tok_text = []
        for sentence in self.tok_sentences:
            for token in sentence:
                self.tok_text.append(token)

        self.lem_text = []
        for sentence in self.lem_sentences:
            for token in sentence:
                self.lem_text.append(token)

        # align sentences
        self.aligned_sentences = []
        for i in range(len(self.raw_sentences)):
            self.aligned_sentences.append((self.raw_sentences[i],
                                           self.tok_sentences[i],
                                           self.lem_sentences[i]))


class F1(object):

    def __call__(self,
                 relevant_term_sequence,
                 tokenized_sentence,
                 likelihoods,
                 syllabicator):
        '''sum of rel term likelihoods / number of syls
        '''
        current_score = 0.
        for term in relevant_term_sequence:
            current_score += likelihoods[term]
        # total syllables
        total_syllables = 0.
        for term in tokenized_sentence:
            total_syllables += len(syllabicator(term))
        return current_score / total_syllables

class F2(object):

    def __call__(self,
                 relevant_term_sequence,
                 tokenized_sentence,
                 relevance_table,
                 syllabicator):
        '''(sum of rel term lklihoods * lambda / number of terms) - syllable entropy
        '''
        current_score = 0.
        for term in relevant_term_sequence:
            current_score += relevance_table[term]

        current_score /= len(tokenized_sentence)
        syllable_entropy = syllable_metric_entropy(tokenized_sentence, syllabicator)
        return current_score - syllable_entropy

class F3(object):

    def __call__(self,
                 relevant_term_sequence,
                 tokenized_sentence,
                 lemmatized_sentence,
                 relevance_table,
                 idf,
                 syllabicator):
        '''(sum of rel term tfidf scores * lambda) / number of syllables
        '''
        current_score = 0.
        for term in relevant_term_sequence:
            tfidf_score = calculate_tf(term, lemmatized_sentence) * idf[term]
            current_score += tfidf_score * relevance_table[term]
        # total syllables
        total_syllables = 0.
        for term in tokenized_sentence:
            total_syllables += len(syllabicator(term))
        return current_score / total_syllables

def clean_working_memory():
    try:
        for f_path in os.listdir(PARENT_DIR / "data/temp/"):
            os.remove(PARENT_DIR / "data/temp/" + f_path)
    except:
        pass

def summarizer(D_path, f_method, seq_method, lemmatizer, session_id=1):

    # Try creating working memory
    try:
        os.makedirs(PARENT_DIR / "data/temp/")
    except:
        pass

    if f_method == "f1":
        f = F1()
    elif f_method == "f2":
        f = F2()
    elif f_method == "f3":
        f = F3()
    else:
        raise Exception("Invalid relevance function")

    if seq_method == "partial":
        seq = overlapping_tokens
    elif seq_method == "probabilistic":
        seq = probabilistic_tokens
    elif seq_method == "lcs":
        seq = lcs_tokens
    else:
        raise Exception("Invalid relevance term sequence calculation method")

    D = Document(D_path, lemmatizer)

    if os.path.exists(PARENT_DIR / ("data/temp/running_summary_%d.pickle" % session_id)):
        with open(PARENT_DIR / ("data/temp/running_summary_%d.pickle" % session_id), "rb") as fp:
            RS = pickle.load(fp)
    else:
        RS = None

    if os.path.exists(PARENT_DIR / ("data/temp/lookup_table_%d.pickle" % session_id)):
        with open(PARENT_DIR / ("data/temp/lookup_table_%d.pickle" % session_id), "rb") as fp:
            lookup_table = pickle.load(fp)
    else:
        lookup_table = {}

    if RS is None:
        RS = D
        new_RS_sent_scores = None
    else:
        # Calculate idf for tfidf, preemptively
        idf = calculate_idf(RS.lem_sentences + D.lem_sentences)

        # Algorithm starts properly
        sequence_of_rt, index_pairs = seq(RS.lem_text,
                                          D.lem_text)

        index_to_sent_in_RS = create_index(RS.aligned_sentences)
        index_to_sent_in_D = create_index(D.aligned_sentences)

        new_RS = []
        new_RS_sent_scores = []

        for term in sequence_of_rt:
            # Log terms
            lookup_table[term] = lookup_table.get(term, 0) + 1
        likelihoods = get_likelihoods(lookup_table)

        while len(sequence_of_rt) > 0:
            term = sequence_of_rt.pop(0)
            term_index_in_RS, term_index_in_D = index_pairs.pop(0)

            # Recover sentences
            candidate_in_RS = index_to_sent_in_RS[term_index_in_RS]
            candidate_in_D = index_to_sent_in_D[term_index_in_D]

            if f_method == "f1":
                score_for_RS_candidate = f(sequence_of_rt,
                                           candidate_in_RS[2],
                                           likelihoods,
                                           RS._syllabicator
                                          )
                score_for_D_candidate = f(sequence_of_rt,
                                          candidate_in_D[2],
                                          likelihoods,
                                          D._syllabicator
                                         )
            elif f_method == "f2":
                score_for_RS_candidate = f(sequence_of_rt,
                                           candidate_in_RS[2],
                                           lookup_table,
                                           RS._syllabicator
                                          )
                score_for_D_candidate = f(sequence_of_rt,
                                          candidate_in_D[2],
                                          lookup_table,
                                          D._syllabicator
                                         )
            elif f_method == "f3":
                score_for_RS_candidate = f(sequence_of_rt,
                                           candidate_in_RS[2],
                                           candidate_in_RS[3],
                                           lookup_table,
                                           idf,
                                           RS._syllabicator
                                          )
                score_for_D_candidate = f(sequence_of_rt,
                                          candidate_in_D[2],
                                          candidate_in_D[3],
                                          lookup_table,
                                          idf,
                                          D._syllabicator
                                         )

            if score_for_RS_candidate > score_for_D_candidate:
                new_RS.append(candidate_in_RS)
                new_RS_sent_scores.append(score_for_RS_candidate)
                # remove other terms from the sequence if included in the
                # candidate
                for_is_finished = False
                breaked = False
                while not for_is_finished:
                    for i in range(len(sequence_of_rt)):
                        future_term_index_in_RS = index_pairs[i][0]
                        future_candidate_in_RS =\
                            index_to_sent_in_RS[future_term_index_in_RS]
                        if future_candidate_in_RS[0] == candidate_in_RS[0]:
                            # Same sentence --> remove
                            del sequence_of_rt[i]
                            del index_pairs[i]
                            breaked = True
                            break
                    if not breaked:
                        for_is_finished = True
                    else:
                        breaked = False
            else:
                new_RS.append(candidate_in_D)
                new_RS_sent_scores.append(score_for_D_candidate)
                # remove other terms from the sequence if included in the
                # candidate
                for_is_finished = False
                breaked = False
                while not for_is_finished:
                    for i in range(len(sequence_of_rt)):
                        future_term_index_in_D = index_pairs[i][1]
                        future_candidate_in_D =\
                            index_to_sent_in_D[future_term_index_in_D]
                        if future_candidate_in_D[0] == candidate_in_D[0]:
                            # Same sentence --> remove
                            del sequence_of_rt[i]
                            del index_pairs[i]
                            breaked = True
                            break
                    if not breaked:
                        for_is_finished = True
                    else:
                        breaked = False

        new_RS_raw_sentences = []
        for _, raw_sentence, _, _ in new_RS:
            new_RS_raw_sentences.append(raw_sentence)

        # Replace old RS
        with open(PARENT_DIR / ("data/temp/running_summary_%d.txt" % session_id), "w") as fp:
            fp.write("\n".join(new_RS_raw_sentences))
        RS = Document(PARENT_DIR / ("data/temp/running_summary_%d.txt" % session_id), lemmatizer)

    with open(PARENT_DIR / ("data/temp/running_summary_%d.pickle" % session_id), "wb") as fp:
        pickle.dump(RS, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(PARENT_DIR / ("data/temp/lookup_table_%d.pickle" % session_id), "wb") as fp:
        pickle.dump(lookup_table, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return RS, new_RS_sent_scores

def summary_limit(summary_sentences, sentence_scores, byte_limit):
    limited_summary = []
    current_length = 0
    score_position = [(i,score) for i, score in enumerate(sentence_scores)]
    score_position.sort(key=lambda x:x[1], reverse=True)

    for position, score in score_position:
        raw_sent, _, _ = summary_sentences[position]
        if (current_length + len(raw_sent.encode("utf-8"))) <= byte_limit:
            limited_summary.append(position)
            current_length += len(raw_sent.encode("utf-8"))
    limited_summary.sort()
    limited_summary = [summary_sentences[pos] for pos in limited_summary]
    return limited_summary

def summary_wordlimit(summary_sentences, sentence_scores, word_limit):
    limited_summary = []
    current_length = 0
    score_position = [(i,score) for i, score in enumerate(sentence_scores)]
    score_position.sort(key=lambda x:x[1], reverse=True)

    for position, score in score_position:
        raw_sent, _, _ = summary_sentences[position]
        if (current_length + raw_sent.count(" ") + 1) <= word_limit:
            limited_summary.append(position)
            current_length += raw_sent.count(" ") + 1
    limited_summary.sort()
    limited_summary = [summary_sentences[pos] for pos in limited_summary]
    return limited_summary
