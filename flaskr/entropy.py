from math import log
from statistics import mean, stdev, variance
from collections import Counter

def shannon_entropy(group):
    total_members = len(group)
    entropy = 0

    frequencies = Counter(group)
    for unique_class in frequencies:
        likelihood = frequencies[unique_class] / total_members
        entropy += likelihood * log(likelihood, 2)

    return (-1) * entropy

def normalized_shannon_entropy(group):
    return shannon_entropy(group) / len(group)

def kullback_leibler_divergence(group_x, group_y):
    # - sum_i p(i)*log( q(i)/p(i) ,2)
    entropy = 0

    likelihood_x = {}
    likelihood_y = {}

    frequencies_x = Counter(group_x)
    total_x = len(group_x) + len(set(group_x))
    for unique_class in frequencies_x:
        likelihood_x[unique_class] = (frequencies_x[unique_class] + 1) / total_x

    frequencies_y = Counter(group_y)
    total_y = len(group_y) + len(set(group_y))
    for unique_class in frequencies_y:
        likelihood_y[unique_class] = (frequencies_y[unique_class] + 1) / total_y

    common_space = set(group_x + group_y)
    for member in common_space:
        p_i = likelihood_x.get(member, 1/total_x)
        q_i = likelihood_y.get(member, 1/total_y)
        entropy += p_i * log(q_i/p_i, 2)
    return (-1) * entropy

def joint_shannon_entropy(group_x, group_y):
    # sum_xy p(x,y)*log( p(x)/p(x,y) ,2)

    entropy = 0

    likelihood_x = {}
    likelihood_y = {}

    frequencies_x = Counter(group_x)
    for unique_class in frequencies_x:
        likelihood_x[unique_class] = frequencies_x[unique_class] / len(group_x)

    frequencies_y = Counter(group_y)
    for unique_class in frequencies_y:
        likelihood_y[unique_class] = frequencies_y[unique_class] / len(group_y)

    for x in likelihood_x:
        p_x = likelihood_x[x]
        for y in likelihood_y:
            p_x_y = p_x * likelihood_y[y]
            entropy += p_x_y * log(p_x_y, 2)
    return (-1) * entropy

def conditional_entropy(group_y, group_x):
    # H(Y|X) = H(X, Y) - H(X)
    joint_entropy = joint_shannon_entropy(group_x, group_y)
    x_entropy = shannon_entropy(group_x)
    return joint_entropy - x_entropy

def mutual_information(group_x, group_y):
    # I(X;Y) = H(X) + H(Y) - H(X, Y)
    x_entropy = shannon_entropy(group_x)
    y_entropy = shannon_entropy(group_y)
    joint_entropy = joint_shannon_entropy(group_x, group_y)
    return x_entropy + y_entropy - joint_entropy

def information_gain(group_x, group_y):
    # IG(X;Y) = H(X) - H(X|Y)
    x_entropy = shannon_entropy(group_x)
    x_given_y_entropy = conditional_entropy(group_x, group_y)
    return x_entropy - x_given_y_entropy

def relevant_term_probabilities(relevant_terms):
    counts = Counter(relevant_terms)
    probabilities = {}
    for term in counts:
        probabilities[term] = counts[term] / len(relevant_terms)
    return probabilities

def syllable_metric_entropy(sentence, syllabicator):
    syllables = []
    for token in sentence:
        token_syllables = syllabicator(token)
        if len(token_syllables) == 0:
            token_syllables = [token]
        syllables += token_syllables

    try:
        syllable_entropy = metric_shannon_entropy(syllables)
    except:
        syllable_entropy = 1.0

    return syllable_entropy

