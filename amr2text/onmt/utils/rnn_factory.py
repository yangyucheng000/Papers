"""
 RNN tools
"""
from __future__ import division

import onmt.models
import mindspore.nn as nn

def rnn_factory(rnn_type, **kwargs):
    """ rnn factory, Use pytorch version when available. """
    no_pack_padded_seq = False
    if rnn_type == "SRU":
        print("enter SRU")
        # SRU doesn't support PackedSequence.
        no_pack_padded_seq = True
        rnn = onmt.models.sru.SRU(**kwargs)
    else:
        print("enter no SRU")
        rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn, no_pack_padded_seq
