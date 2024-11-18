#!/usr/bin/env python3

# Subclass ConditionalRandomFieldBackprop to get a model that uses some
# contextual features of your choice.  This lets you test the revision to hmm.py
# that uses those features.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import tensor, Tensor, cuda
from jaxtyping import Float

from corpus import Tag, Word
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop, TorchScalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomFieldTest(ConditionalRandomFieldBackprop):
    """A CRF with some arbitrary non-stationary features, for testing."""
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False):
        """Construct an CRF with initially random parameters, with the
        given tagset, vocabulary, and lexical features.  See the super()
        method for discussion."""

        # an __init__() call to the nn.Module class must be made before assignment on the child.
        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)

        self.E = lexicon          # rows are word embeddings
        self.e = lexicon.size(1)  # dimensionality of word embeddings
        self.rnn_dim = rnn_dim

    @override
    def init_params(self) -> None:
        # [docstring will be inherited from parent method]

        raise NotImplementedError   # you fill this in!

    @override
    def updateAB(self) -> None:
        # Nothing to do - self.A and self.B are not used in non-stationary CRFs
        pass

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
        # [docstring will be inherited from parent method]

        # You need to override this function to compute your non-stationary features.

        raise NotImplementedError   # you fill this in!

        return self.A + non_stationary_features_A   # example
        
        
    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
        # [docstring will be inherited from parent method]

        raise NotImplementedError   # you fill this in!

        return self.B + non_stationary_features_B    # example
