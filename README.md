# Multilang Summarizer

This package implements an online multi-document summarization algorithm, intended to improve text readability. It supports the following languages:

* 'de': 'German'
* 'en': 'English'
* 'es': 'Spanish'
* 'fr': 'French'
* 'hu': 'Hungarian'
* 'it': 'Italian'
* 'pt': 'Portuguese'
* 'ro': 'Romanian'
* 'sv': 'Swedish'

This work was partially supported by the National Council of Science and Technology (CONACYT) of Mexico, as part of the Cátedras CONACYT project _Infraestructura para agilizar el desarrollo de sistemas centrados en el usuario_, Ref. 3053.

### Prerequisites

This projects has the following dependencies:

* Pyphen
* TextStat
* sentence-splitter
* numpy
* NLTK (needs to download tokenization corpora)

### Installing

The package is distributed via pip:

```
pip install multilang-summarizer
```

### Use

The summarizer function directly implements the algorithm.

```
from multilang_summarizer.summarizer import summarizer

# summarizer(D_path, f_method, seq_method, lemmatizer, session_id=1)
```

It receives the path to a single document, a choice for three different sentence relevance functions (f\_method), a relevant term selection method (seq\_method), a lemmatizer and a session number (for memory purposes).

The choice of f\_method can be one of three:

* 'f1' : uses mean term likelihood as an indicator of relevance.
* 'f2' : uses past term use and syllabic entropy to measure relevance and sentence complexity, respectively.
* 'f3' : uses a weighted tfidf-based approach to measure relevance.

The choice of seq\_method can be one of three:

* 'partial' : uses simple matching between the last generated summary and the new input to identify relevant terms.
* 'probability' : uses past term likelihoods to identify relevant terms.
* 'lcs' : uses the Longest Common Subsequence algorithm to identify relevant terms between the last generated summary and the new input.

The _lemmatizer_ object contains the lemmatization rules for the selected language. For English, it can be instanced as follows:

```
from multilang_summarizer.lemmatizer import Lemmatizer


lemmatizer = Lemmatizer.for_language("en")
```

Finally, session\_id tells the algorithm to which running summary input D will be adding to. Different sessions can be opened at once. To clean the cache __for all sessions__ use the following method:

```
from multilang_summarizer.summarizer import clean_working_memory

clean_working_memory()
```

In the end, summarizer returns a Document object containing all the sentences selected from all previous documents in the named session, and the _f_ score with which each sentence was selected.

## Running the tests

Two example scripts are provided in the repo:

* tests/test\_english.py
* tests/test\_spanish.py

To run them, the documents in the test\_documents folder are required. Simply, execute

```
python tests/test_english.py
``` 

from the root folder after setup.

### Example results

The following summary was obtained using f\_method = 'f3' and seq\_method = 'lcs' over the 10 news items in the test\_documents folder.

```
For the second day in a row, astronauts boarded space shuttle Endeavour 
on Friday for liftoff on NASA's first space station construction flight.
The decision, which followed ``frank and 
candid'' discussions between the two partners, was not imposed by 
the United States, he said.
The main cargo Thursday was the Unity module, the first U.S.-built 
station part.
The shuttle contains 
the second station component.
The mechanical arm has never before moved anything so big.
The bigger worry, by far, was over Endeavour's pursuit and capture 
of Zarya, and its coupling with Unity.
```

In Spanish, the following summary was obtained using f\_method = 'f1' and seq\_method = 'lcs' over the 12 Spanish-language news items in the test\_documents folder.

```
Tras una intensa búsqueda llevada a cabo por rescatistas, los 12 niños y su profesor fueron encontrados con vida y en buen estado de salud.
El rescate de los 12 niños y su entrenador que quedaron atrapados en una cueva inundada, en el norte de Tailandia, podría tomar semanas o incluso meses.
Pero aunque los 13 pudieran bucear, algunas partes de la cueva son demasiado estrechas,lo que exige mucho entrenamiento para poder pasar con tanques de buceo.
Los niños fueron encontrados, 200 metros más adelante.
Están cansados y necesitan un tiempo para reponerse.
La primera etapa del rescate es hacerles recuperar fuerzas.
Los 13 miembros están bien.
```
## Authors

* **Arturo Curiel** - [arturocuriel.com](https://www.arturocuriel.com/)

## License

This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Thanks to [Claudio Gutierrez Soto](http://www.face.ubiobio.cl/~cgutierr/) and [Rafael Rojano](https://scholar.google.com/citations?user=tJO7AnxQtZUC&hl=en) for their input in the development on the algorithm.

