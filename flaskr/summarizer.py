import copy
import os
import sys

from collections import Counter

from functools import partial

from math import log

from flaskr.lcs import *

from flaskr.tfidf import calculate_tf, calculate_idf
from flaskr.entropy import sentence_syllable_metric_entropy, conditional_entropy

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

def create_index(sents_w_tokenization):
    index = {}
    start = 0
    for sent, tokenized_sent in sents_w_tokenization:
        for i in range(start, start + len(tokenized_sent)):
            index[i] = (sent, tokenized_sent)
        start += len(tokenized_sent)
    return index

def overlapping_summary(scoring,
                        sents_w_tokenization_1, tokens_1,
                        sents_w_tokenization_2, tokens_2,
                        language="en", additions="none",
                        previous_overlap_terms_index={}):
    '''Returns one summary with sentences from both docs
    previous_lcs_terms_index gives an additional value to terms'''

    entropy_calculation = partial(sentence_syllable_metric_entropy,
                                  language)
    sylcount_calculation = partial(count_syllables,
                                   language)

    if additions != 'none':
        if additions == "tfidf" :
            with_tfidf = True
        elif additions == "entropy" :
            with_entropy = True
        elif additions == "readability" :
            with_readability = True
        elif additions == "inverse_readability" :
            with_inverse_readability = True
        elif additions == "tfidf_readability" :
            with_tfidf = True
            with_readability = True
        elif additions == "tfidf_inverse_readability" :
            with_tfidf = True
            with_inverse_readability = True
        else:
            raise ValueError("Unknown Scoring function %s" % additions)

        if with_tfidf:
            # Get idf considering all sentences (both docs)
            idf = calculate_idf(
                [doc[1] for doc in sents_w_tokenization_1 + sents_w_tokenization_2]
                    )

    if scoring == "zero":
        overlap_result, overlap_result_indexes = overlapping_tokens(tokens_1, tokens_2)
    else:
        overlap_result, overlap_result_indexes = probabilistic_tokens(tokens_1, tokens_2)

    index1 = create_index(sents_w_tokenization_1)
    index2 = create_index(sents_w_tokenization_2)

    overlap_probabilities = {}
    # New weights
    for term in overlap_result:
        if term not in previous_overlap_terms_index.keys():
            previous_overlap_terms_index[term] = 1
        else:
            previous_overlap_terms_index[term] += 1
        previous_overlap_terms_index["__len__"] = 1 +\
                                previous_overlap_terms_index.get("__len__", 0)
        overlap_probabilities[term] = previous_overlap_terms_index[term] /\
                                      previous_overlap_terms_index["__len__"]

    assert(len(overlap_result) > 0)
    # Count overlapping tokens
    summary = []
    summary_sentence_scores = []
    # Calculate candidates
    for i, j in overlap_result_indexes:
        # Get candidate pairs
        if index1[i] not in summary:
            raw_tokenization_1 = raw_tokenize(index1[i][0], lang_stopwords)
            syllable_count_1 = sylcount_calculation(raw_tokenization_1)
            if index2[j] not in summary:
                raw_tokenization_2 = raw_tokenize(index2[j][0], lang_stopwords)
                syllable_count_2 = sylcount_calculation(raw_tokenization_2)

                # Sentence from first doc
                if scoring == "zero":
                    overlaps, _ = overlapping_tokens(overlap_result, index1[i][1])
                else:
                    overlaps, _ = probabilistic_tokens(overlap_result, index1[i][1])
                overlap_value = 0
                if with_tfidf:
                    for term in overlaps:
                        term_weight = calculate_tf(term, index1[i][1]) * idf[term]
                        overlap_value += term_weight *\
                                         previous_overlap_terms_index[term]
                    #score1 = overlap_value
                    score1 = overlap_value / syllable_count_1
                else:
                    for term in overlaps:
                        if not with_entropy:
                            overlap_value += overlap_probabilities[term]
                        else:
                            overlap_value += previous_overlap_terms_index[term]
                    if not with_entropy:
                        score1 = overlap_value / syllable_count_1
                    else:
                        score1 = overlap_value / len(index1[i][1])

                if with_entropy:
                    score1 -= entropy_calculation([raw_tokenization_1,
                                                   index1[i][1]])
                elif with_readability:
                    #score1 += 2 * readability_measure([index1[i][1]])
                    try:
                        score1 = score1/len(overlaps) + readability_measure([index1[i][1]])
                    except:
                        score1 = 0
                elif with_inverse_readability:
                    #score1 += 2 * invert(readability_measure, [index1[i][1]])
                    try:
                        score1 = score1/len(overlaps) + invert(readability_measure, [index1[i][1]])
                    except:
                        score1 = 0

                # Sentence from second doc
                if scoring == "zero":
                    overlaps, _ = overlapping_tokens(overlap_result, index2[j][1])
                else:
                    overlaps, _ = probabilistic_tokens(overlap_result, index2[j][1])
                overlap_value = 0
                if with_tfidf:
                    for term in overlaps:
                        term_weight = calculate_tf(term, index2[j][1]) * idf[term]
                        overlap_value += term_weight *\
                                         previous_overlap_terms_index[term]
                    #score2 = overlap_value
                    score2 = overlap_value / syllable_count_2
                else:
                    for term in overlaps:
                        if not with_entropy:
                            overlap_value += overlap_probabilities[term]
                        else:
                            overlap_value += previous_overlap_terms_index[term]
                    if not with_entropy:
                        score2 = overlap_value / syllable_count_2
                    else:
                        score2 = overlap_value / len(index2[j][1])

                if with_entropy:
                    score2 -= entropy_calculation([raw_tokenization_2,
                                                   index2[j][1]])
                elif with_readability:
                    #score2 += 2 * readability_measure([index2[j][1]])
                    try:
                        score2 = score2/len(overlaps) + readability_measure([index2[j][1]])
                    except:
                        score2 = 0
                elif with_inverse_readability:
                    #score2 += 2 * invert(readability_measure, [index2[j][1]])
                    try:
                        score2 = score2/len(overlaps) + invert(readability_measure, [index2[j][1]])
                    except:
                        score2 = 0

                if score1 >= score2:
                    summary.append(index1[i])
                    summary_sentence_scores.append(score1)
                else:
                    summary.append(index2[j])
                    summary_sentence_scores.append(score2)
    return summary, overlap_result, previous_overlap_terms_index,\
            summary_sentence_scores

def mixed_summary_indexes(sents_w_tokenization_1, tokens_1,
                          sents_w_tokenization_2, tokens_2,
                          language="en", additions="none",
                          previous_lcs_terms_index={}):
    '''Returns one summary with sentences from both docs
    previous_lcs_terms_index gives an additional value to terms'''

    with_tfidf = False
    with_entropy = False
    with_readability = False
    with_inverse_readability = False

    readability_measure = None
    if language == "en":
        readability_measure = flesch_kincaid
        lang_stopwords = en_stopwords
    elif language == "es":
        readability_measure = szigriszt_pazos
        lang_stopwords = es_stopwords
    else:
        raise ValueError("Not english nor spanish")

    entropy_calculation = partial(sentence_syllable_metric_entropy,
                                  language)
    sylcount_calculation = partial(count_syllables,
                                   language)

    if additions != 'none':
        if additions == "tfidf" :
            with_tfidf = True
        elif additions == "entropy" :
            with_entropy = True
        elif additions == "readability" :
            with_readability = True
        elif additions == "inverse_readability" :
            with_inverse_readability = True
        elif additions == "tfidf_readability" :
            with_tfidf = True
            with_readability = True
        elif additions == "tfidf_inverse_readability" :
            with_tfidf = True
            with_inverse_readability = True
        else:
            raise ValueError("Unknown Scoring function %s" % additions)

        if with_tfidf:
            # Get idf considering all sentences (both docs)
            idf = calculate_idf(
                [doc[1] for doc in sents_w_tokenization_1 + sents_w_tokenization_2]
                    )
    lcs_result = lcs(tokens_1, tokens_2)
    lcs_result_indexes = lcs_indexes(tokens_1, tokens_2)

    index1 = create_index(sents_w_tokenization_1)
    index2 = create_index(sents_w_tokenization_2)

    overlap_probabilities = {}
    # New weights
    for term in lcs_result:
        if term not in previous_lcs_terms_index.keys():
            previous_lcs_terms_index[term] = 1
        else:
            previous_lcs_terms_index[term] += 1
        previous_lcs_terms_index["__len__"] = 1 +\
                                previous_lcs_terms_index.get("__len__", 0)
        overlap_probabilities[term] = previous_lcs_terms_index[term] /\
                                      previous_lcs_terms_index["__len__"]

    doc1_candidates = []
    doc1_candidate_scores = []
    doc1_candidate_indexes = []
    step = []
    for i, _ in lcs_result_indexes:
        if index1[i] not in doc1_candidates:
            raw_tokenization_1 = raw_tokenize(index1[i][0], lang_stopwords)
            syllable_count_1 = sylcount_calculation(raw_tokenization_1)

            phrase_lcs = lcs(lcs_result, index1[i][1])
            lcs_value = 0

            if with_tfidf:
                for term in phrase_lcs:
                    term_weight = calculate_tf(term, index1[i][1]) * idf[term]
                    lcs_value += term_weight *\
                                 previous_lcs_terms_index[term]
                #score = lcs_value
                score = lcs_value / syllable_count_1
            else:
                for term in phrase_lcs:
                    if not with_entropy:
                        lcs_value += overlap_probabilities[term]
                    else:
                        lcs_value += previous_lcs_terms_index[term]
                if not with_entropy:
                    score = lcs_value / syllable_count_1
                else:
                    score = lcs_value / len(index1[i][1])

            if with_entropy:
                score -= entropy_calculation([raw_tokenization_1,
                                              index1[i][1]])
            elif with_readability:
                #score += 2 * readability_measure([index1[i][1]])
                score = score/len(phrase_lcs) + readability_measure([index1[i][1]])
            elif with_inverse_readability:
                #score += 2 * invert(readability_measure, [index1[i][1]])
                score = score/len(phrase_lcs) + invert(readability_measure, [index1[i][1]])
            doc1_candidate_scores.append(score)
            doc1_candidates.append(index1[i])
            if len(step) > 0 :
                doc1_candidate_indexes.append(step)
            step = [i]
        else:
            step.append(i)
    doc1_candidate_indexes.append(step)

    doc2_candidates = []
    doc2_candidate_scores = [ ]
    doc2_candidate_indexes = []
    step = []
    for _, j in lcs_result_indexes:
        if index2[j] not in doc2_candidates:
            raw_tokenization_2 = raw_tokenize(index2[j][0], lang_stopwords)
            syllable_count_2 = sylcount_calculation(raw_tokenization_2)

            phrase_lcs = lcs(lcs_result, index2[j][1])
            lcs_value = 0

            if with_tfidf:
                for term in phrase_lcs:
                    term_weight = calculate_tf(term, index2[j][1]) * idf[term]
                    lcs_value += term_weight *\
                                 previous_lcs_terms_index[term]
                #score = lcs_value
                score = lcs_value / syllable_count_2
            else:
                for term in phrase_lcs:
                    if not with_entropy:
                        lcs_value += overlap_probabilities[term]
                    else:
                        lcs_value += previous_lcs_terms_index[term]
                if not with_entropy:
                    score = lcs_value / syllable_count_2
                else:
                    score = lcs_value / len(index2[j][1])

            if with_entropy:
                score -= entropy_calculation([raw_tokenization_2,
                                              index2[j][1]])
            elif with_readability:
                #score += 2 * readability_measure([index2[j][1]])
                score = score/len(phrase_lcs) + readability_measure([index2[j][1]])
            elif with_inverse_readability:
                #score += 2 * invert(readability_measure, [index2[j][1]])
                score = score/len(phrase_lcs) + invert(readability_measure, [index2[j][1]])

            doc2_candidate_scores.append(score)
            doc2_candidates.append(index2[j])
            if len(step) > 0 :
                doc2_candidate_indexes.append(step)
            step = [j]
        else:
            step.append(i)
    doc2_candidate_indexes.append(step)

    assert(len(doc1_candidates) > 0)
    assert(len(doc2_candidates) > 0)

    # Calculate summary
    modifiable_lcs_result = lcs_result.copy()
    summary = []
    summary_sentence_scores = []

    doc1_candidate = doc1_candidates.pop(0)
    doc1_score = doc1_candidate_scores.pop(0)
    doc1_indexes = doc1_candidate_indexes.pop(0)

    doc2_candidate = doc2_candidates.pop(0)
    doc2_score = doc2_candidate_scores.pop(0)
    doc2_indexes = doc2_candidate_indexes.pop(0)
    while len(modifiable_lcs_result) > 0:
        # The sentence of 1 has larger score thant 2
        if doc1_score >= doc2_score:
            # Add to the summary
            summary.append(doc1_candidate)
            summary_sentence_scores.append(doc1_score)
            # Check how many LCS terms we have covered
            # and remove them from the running sequence
            total_to_remove = len(doc1_indexes)
            del modifiable_lcs_result[:total_to_remove]
            # The current doc1 sentences was added to the summary, get 
            # the next one
            try:
                doc1_candidate = doc1_candidates.pop(0)
                doc1_score = doc1_candidate_scores.pop(0)
                doc1_indexes = doc1_candidate_indexes.pop(0)
            except:
                assert(len(modifiable_lcs_result) == 0)
            # Stop considering them in document2
            while total_to_remove > 0:
                doc2_indexes.pop(0)
                total_to_remove -= 1
                # The sentences has no LCS terms anymore, consider
                # the next one
                if len(doc2_indexes) == 0:
                    try:
                       doc2_candidate = doc2_candidates.pop(0)
                       doc2_score = doc2_candidate_scores.pop(0)
                       doc2_indexes = doc2_candidate_indexes.pop(0)
                    except:
                        assert(len(modifiable_lcs_result) == 0)
            # Recalculate doc2 score considering the corrected qty of lcs terms
            raw_tokenization_2 = raw_tokenize(doc2_candidate[0], lang_stopwords)
            syllable_count_2 = sylcount_calculation(raw_tokenization_2)

            lcs_value = 0
            if with_tfidf:
                for i in range(len(doc2_indexes)):
                    term_weight = calculate_tf(modifiable_lcs_result[i],
                            doc2_candidate[1]) * idf[modifiable_lcs_result[i]]
                    lcs_value += term_weight *\
                                 previous_lcs_terms_index[modifiable_lcs_result[i]]
                #doc2_score = lcs_value
                doc2_score = lcs_value / syllable_count_2
            else:
                for i in range(len(doc2_indexes)):
                    if not with_entropy:
                        lcs_value += overlap_probabilities[modifiable_lcs_result[i]]
                    else:
                        lcs_value += previous_lcs_terms_index[modifiable_lcs_result[i]]
                if not with_entropy:
                    doc2_score = lcs_value / syllable_count_2
                else:
                    doc2_score = lcs_value / len(doc2_candidate[1])
            if with_entropy:
                doc2_score -= entropy_calculation([raw_tokenization_2,
                                                   doc2_candidate[1]])
            elif with_readability:
                #doc2_score += 2 * readability_measure([doc2_candidate[1]])
                try:
                    doc2_score = doc2_score/len(doc2_indexes) +\
                            readability_measure([doc2_candidate[1]])
                except:
                    doc2_score = 0
            elif with_inverse_readability:
                #doc2_score += 2 * invert(readability_measure,
                #                         [doc2_candidate[1]])
                try:
                    doc2_score = doc2_score/len(doc2_indexes) + invert(readability_measure,
                                                                       [doc2_candidate[1]])
                except:
                    doc2_score = 0
        else: # doc2 sentence scores higher than doc1 sentence
            # Add to the summary
            summary.append(doc2_candidate)
            summary_sentence_scores.append(doc2_score)
            # Check how many LCS terms we have covered
            # and remove them from the running sequence
            total_to_remove = len(doc2_indexes)
            del modifiable_lcs_result[:total_to_remove]
            # The current doc2 sentences was added to the summary, get 
            # the next one
            try:
               doc2_candidate = doc2_candidates.pop(0)
               doc2_score = doc2_candidate_scores.pop(0)
               doc2_indexes = doc2_candidate_indexes.pop(0)
            except:
                assert(len(modifiable_lcs_result) == 0)
            # Stop considering them in document1
            while total_to_remove > 0:
                doc1_indexes.pop(0)
                total_to_remove -= 1
                # The sentences has no LCS terms anymore, consider
                # the next one
                if len(doc1_indexes) == 0:
                    try:
                        doc1_candidate = doc1_candidates.pop(0)
                        doc1_score = doc1_candidate_scores.pop(0)
                        doc1_indexes = doc1_candidate_indexes.pop(0)
                    except:
                        assert(len(modifiable_lcs_result) == 0)
            # Recalculate doc1 score considering the corrected qty of lcs terms
            raw_tokenization_1 = raw_tokenize(doc1_candidate[0], lang_stopwords)
            syllable_count_1 = sylcount_calculation(raw_tokenization_1)

            lcs_value = 0
            if with_tfidf:
                for i in range(len(doc1_indexes)):
                    term_weight = calculate_tf(modifiable_lcs_result[i],
                            doc1_candidate[1]) * idf[modifiable_lcs_result[i]]
                    lcs_value += term_weight *\
                                 previous_lcs_terms_index[modifiable_lcs_result[i]]
                #doc1_score = lcs_value
                doc1_score = lcs_value / syllable_count_1
            else:
                for i in range(len(doc1_indexes)):
                    if not with_entropy:
                        lcs_value += overlap_probabilities[modifiable_lcs_result[i]]
                    else:
                        lcs_value += previous_lcs_terms_index[modifiable_lcs_result[i]]
                if not with_entropy:
                    doc1_score = lcs_value / syllable_count_1
                else:
                    doc1_score = lcs_value / len(doc1_candidate[1])

            if with_entropy:
                doc1_score -= entropy_calculation([raw_tokenization_1,
                                                   doc1_candidate[1]])
            elif with_readability:
                #doc1_score += 2 * readability_measure([doc1_candidate[1]])
                try:
                    doc1_score = doc1_score/len(doc1_indexes) + readability_measure([doc1_candidate[1]])
                except:
                    doc1_score = 0
            elif with_inverse_readability:
                #doc1_score += 2 * invert(readability_measure,
                #                         [doc1_candidate[1]])
                try:
                    doc1_score = doc1_score/len(doc1_indexes) + invert(readability_measure,
                                                                       [doc1_candidate[1]])
                except:
                    doc1_score = 0

    return summary, lcs_result, previous_lcs_terms_index,\
            summary_sentence_scores

def summary_limit(summary_sentences, sentence_scores, byte_limit):
    limited_summary = []
    current_length = 0
    score_position = [(i,score) for i, score in enumerate(sentence_scores)]
    score_position.sort(key=lambda x:x[1], reverse=True)

    for position, score in score_position:
        raw_sent, _ = summary_sentences[position]
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
        raw_sent, _ = summary_sentences[position]
        if (current_length + raw_sent.count(" ") + 1) <= word_limit:
            limited_summary.append(position)
            current_length += raw_sent.count(" ") + 1
    limited_summary.sort()
    limited_summary = [summary_sentences[pos] for pos in limited_summary]
    return limited_summary
