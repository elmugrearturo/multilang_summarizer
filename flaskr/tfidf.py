from math import log
import numpy as np
import sys

sys.setrecursionlimit(10000)

def flatten(multidim_list):
    if isinstance(multidim_list, list):
        if len(multidim_list) > 1:
            return flatten(multidim_list[0]) + flatten(multidim_list[1:])
        elif len(multidim_list) > 0:
            return flatten(multidim_list[0])
        else:
            return []
    else:
        return [multidim_list]

def calculate_idf(tokenized_documents):
    type_set = list(set(flatten(tokenized_documents)))
    idf = {}
    total_num_of_docs = len(tokenized_documents)
    for word in type_set:
        idf[word] = 0
        for document in tokenized_documents:
            if word in document:
                idf[word] += 1
        idf[word] = log(total_num_of_docs / idf[word])
    return idf

def calculate_tf(word, tokenized_document):
    return tokenized_document.count(word) / len(tokenized_document)

def tfidf_matrix(tokenized_documents):
    idf = calculate_idf(tokenized_documents)
    matrix = []
    terms = sorted(list(idf.keys()))
    for document in tokenized_documents:
        doc_vector = []
        for word in terms:
            doc_vector.append(calculate_tf(word, document) * idf[word])
        matrix.append(doc_vector)
    return matrix, terms, idf

def represent_in_tfidf(tokenized_document, terms, idf):
    new_vector = []
    for term in terms:
        new_vector.append(calculate_tf(term, tokenized_document) * idf[term])
    return np.array(new_vector)

