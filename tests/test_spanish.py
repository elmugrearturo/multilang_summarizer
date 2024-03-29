import pickle
from multilang_summarizer.lemmatizer import Lemmatizer
import xml.etree.ElementTree as ET
from multilang_summarizer.summarizer import Document, clean_working_memory, summarizer, summary_limit, summary_wordlimit

from multilang_summarizer.readability import szigriszt_pazos

import random

import os

test_dir = "./test_documents/es/"
output_dir = "./output_documents/es/"

test_xml = test_dir + "news05 tailandia-ninios.xml"
tree = ET.parse(test_xml)
root = tree.getroot()
current_text = 1
for child in root:
    for subchild in child:
        if subchild.tag == "content":
            path = test_dir + "%d.txt" % current_text
            with open(path, "w") as fp:
                fp.write(subchild.text)
            current_text += 1

# Try to create needed dirs in the beginning
try:
    os.makedirs(output_dir)
except:
    pass

# Cleanup
clean_working_memory()

for f_path in os.listdir(output_dir):
    os.remove(output_dir + f_path)

# Read input paths
paths = []
for f_path in os.listdir(test_dir):
    if f_path.endswith(".txt"):
        paths.append(test_dir + f_path)

random.shuffle(paths)

lemmatizer = Lemmatizer.for_language("es")

RS = {}
scores = {}
for path in paths:
    RS[1], scores[1] = summarizer(path, "f1", "partial", lemmatizer, 21)
    RS[2], scores[2] = summarizer(path, "f1", "probabilistic", lemmatizer, 22)
    RS[3], scores[3] = summarizer(path, "f1", "lcs", lemmatizer, 23)

    RS[4], scores[4] = summarizer(path, "f2", "partial", lemmatizer, 24)
    RS[5], scores[5] = summarizer(path, "f2", "probabilistic", lemmatizer, 25)
    RS[6], scores[6] = summarizer(path, "f2", "lcs", lemmatizer, 26)

    RS[7], scores[7] = summarizer(path, "f3", "partial", lemmatizer, 27)
    RS[8], scores[8] = summarizer(path, "f3", "probabilistic", lemmatizer, 28)
    RS[9], scores[9] = summarizer(path, "f3", "lcs", lemmatizer, 29)

    print("Processed:", path)

byte_limit = 661 # number of words
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
        print("\nReadability", szigriszt_pazos(limited_summary.tok_sentences))
    except:
        pass
