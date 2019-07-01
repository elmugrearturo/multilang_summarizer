import pickle
from multilang_lemmatizer import Lemmatizer
import xml.etree.ElementTree as ET
from multilang_summarizer.summarizer import Document, summarizer, summary_limit, summary_wordlimit

from multilang_summarizer.readability import szigriszt_pazos

import os

test_dir = "./test_documents/"
output_dir = "./output_documents/"
data_dir = "./data/"

#test_xml = "./test_documents/news05 tailandia-ninios.xml"
#tree = ET.parse(test_xml)
#root = tree.getroot()
#current_text = 1
#for child in root:
#    for subchild in child:
#        if subchild.tag == "content":
#            path = "./test_documents/%d.txt" % current_text
#            with open(path, "w") as fp:
#                fp.write(subchild.text)
#            current_text += 1

# Cleanup
for f_path in os.listdir(data_dir):
    os.remove(data_dir + f_path)

paths = []
for f_path in os.listdir(test_dir):
    if f_path.endswith(".txt"):
        paths.append(test_dir + f_path)

lemmatizer = Lemmatizer.for_language("es")

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

byte_limit = 815
for i in range(1, 10):
    limited_summary = summary_limit(RS[i].aligned_sentences, scores[i],
                                    byte_limit)
    raw_limited_summary = "\n".join([raw_sent for raw_sent, _, _ in\
                                     limited_summary])
    with open("./output_documents/limited_summary_%d.txt" % i, "w") as fp:
        fp.write(raw_limited_summary)
    print("\n\n", i, "\n\n", raw_limited_summary)
    limited_summary = Document("./output_documents/limited_summary_%d.txt" %i,
                               lemmatizer)
    print("\nReadability", szigriszt_pazos(limited_summary.tok_sentences))
