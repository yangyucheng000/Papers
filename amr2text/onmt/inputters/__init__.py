"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""
from onmt.inputters.inputter import collect_feature_vocabs, make_features, \
    collect_features, get_num_features, \
    load_fields_from_vocab, get_fields, \
    save_fields_to_vocab, build_dataset, \
    build_vocab, merge_vocabs
from onmt.inputters.dataset_base import PAD_WORD, BOS_WORD, \
    EOS_WORD, UNK


__all__ = ['PAD_WORD', 'BOS_WORD', 'EOS_WORD', 'UNK',
           'collect_feature_vocabs', 'make_features',
           'collect_features', 'get_num_features',
           'load_fields_from_vocab', 'get_fields',
           'save_fields_to_vocab', 'build_dataset',
           'build_vocab', 'merge_vocabs',]
