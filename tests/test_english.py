import pickle
from multilang_summarizer.lemmatizer import Lemmatizer
import xml.etree.ElementTree as ET
from multilang_summarizer.summarizer import Document, summarizer, summary_limit, summary_wordlimit

from multilang_summarizer.readability import flesch_kincaid

import random

import os

test_dir = "./test_documents/en/"
output_dir = "./output_documents/en/"
data_dir = "./data/en/"

current_source = 0
for f_path in os.listdir(test_dir):
    if not f_path.endswith(".txt"):
        document_path = test_dir + f_path
        tree = ET.parse(document_path)
        root = tree.getroot()
        original_text = root.find("TEXT").text
        with open(test_dir + "%d.txt" % current_source, "w") as fp:
            fp.write(original_text)
        current_source += 1

# Try to create needed dirs in the beginning
try:
    os.makedirs(data_dir)
except:
    pass

try:
    os.makedirs(output_dir)
except:
    pass

# Cleanup
for f_path in os.listdir(data_dir):
    os.remove(data_dir + f_path)

for f_path in os.listdir(output_dir):
    os.remove(output_dir + f_path)

# Read input paths
paths = []
for f_path in os.listdir(test_dir):
    if f_path.endswith(".txt"):
        paths.append(test_dir + f_path)

random.shuffle(paths)

lemmatizer = Lemmatizer.for_language("en")

RS = {}
scores = {}
for path in paths:
    RS[1], scores[1] = summarizer(path, "f1", "partial", lemmatizer, 1)
    RS[2], scores[2] = summarizer(path, "f1", "probabilistic", lemmatizer, 2)
    RS[3], scores[3] = summarizer(path, "f1", "lcs", lemmatizer, 3)

    RS[4], scores[4] = summarizer(path, "f2", "partial", lemmatizer, 4)
    RS[5], scores[5] = summarizer(path, "f2", "probabilistic", lemmatizer, 5)
    RS[6], scores[6] = summarizer(path, "f2", "lcs", lemmatizer, 6)

    RS[7], scores[7] = summarizer(path, "f3", "partial", lemmatizer, 7)
    RS[8], scores[8] = summarizer(path, "f3", "probabilistic", lemmatizer, 8)
    RS[9], scores[9] = summarizer(path, "f3", "lcs", lemmatizer, 9)

    print("Processed:", path)

byte_limit = 661 # bytes
for i in range(1, 10):
    limited_summary = summary_limit(RS[i].aligned_sentences, scores[i],
                                    byte_limit)
    raw_limited_summary = "\n".join([raw_sent for raw_sent, _, _ in\
                                     limited_summary])
    with open(output_dir + "limited_summary_%d.txt" % i, "w") as fp:
        fp.write(raw_limited_summary)

    print("\n\n", i, "\n\n", raw_limited_summary)
    limited_summary = Document(output_dir + "limited_summary_%d.txt" %i,
                               lemmatizer)
    try:
        print("\nReadability", flesch_kincaid(limited_summary.tok_sentences))
    except:
        pass
# Cleanup
for f_path in os.listdir(data_dir):
    os.remove(data_dir + f_path)
