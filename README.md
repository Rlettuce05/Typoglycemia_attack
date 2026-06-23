# Typoglycemia_attack

Generate typoglycemia-poisoned captions for research datasets.

The current script only shuffles candidate words tagged as nouns or verbs
(`NN*` or `VB*` POS tags). `Typoglycemia` accepts a custom `pos_tagger`
callable for stricter tagging. When none is supplied, it uses NLTK if available
and otherwise falls back to a conservative built-in tagger.
