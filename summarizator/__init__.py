from lemmatizer import lemma_index, language_index

lemmatizers = {}
for key in language_index:
    lemmatizers[key] = lemma_index(key)

import ipdb;ipdb.set_trace()
