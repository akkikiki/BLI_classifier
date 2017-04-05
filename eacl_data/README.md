# EACL Data
In this folder you find the comparable corpus and corresponding bilingual lexicons that were used in our EACL paper.


Articles:

* **articles.en.txt, articles.nl.txt** contain English/Dutch wikipedia documents in the medical domain. Each document starts with an id and ends the <eod> token.
*	**articles.tok.lower.en.txt, articles.tok.lower.nl.txt**: the preprocessed version articles.en.txt and articles.nl.txt. Every line contains one sentence, tokenization was performed using the Moses tokenizer and all words are lowercased. This files were used to train the Skip-gram embeddings.
*	**articles.ennl.bwesg.txt** : the file that was used to train the BWESG embeddings. Every line corresponds to one document pair. Words are suffixed with their language code _en or _nl. 

Lexicons:

*	**lex.filtered.test80-20.txt**: the bilingual lexicon that is used for testing. We performed a 80% for training/ 20 % for testing split. filtered refers to the fact that translations that can’t be found in the corpus after preprocessing are filtered out of the lexicon.
*	**lex.filtered.train80-20.txt**:  the bilingual lexicon that is used for training. We performed a 80% for training/ 20% for testing split. filtered refers to the fact that translations that can not be found in the corpus after preprocessing are filtered out of the lexicon.
*	**lex.filtered.train80-20.tune.txt**: when doing tuning experiments, use this file for training. It contains 80% of the translations in lex.filtered.train80-20.txt
*	**lex.filtered.validation80-20.txt**: when we doing tuning experiments, we use this file for validation. It contains 20% of the translations in  lex.filtered.train80-20.txt

Embeddings:

* **ennl.mono.dim=50.bin**: embeddings trained with the continuous Skip-gram model. Embeddings are trained monolingually on the English/Dutch articles separately. Then we concatenated the embeddings in a single file, English words were suffixed with _en, Dutch words were suffix with _nl.
* **ennl.bwesg.dim=50.window=100.bin**: embeddings trained with BWESG. This comes down to training embeddings with continuous Skip-gram on the _articles.ennl.bwesg.txt_ file. 