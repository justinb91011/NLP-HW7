#!/usr/bin/env python3

# CS465 at Johns Hopkins University.

# Subclass ConditionalRandomFieldBackprop to get a biRNN-CRF model.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf, log, exp
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import Tensor, cuda
from jaxtyping import Float

from corpus import IntegerizedSentence, Sentence, Tag, TaggedCorpus, Word
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop, TorchScalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomFieldNeural(ConditionalRandomFieldBackprop):
    """A CRF that uses a biRNN to compute non-stationary potential
    matrices.  The feature functions used to compute the potentials
    are now non-stationary, non-linear functions of the biRNN
    parameters."""
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False):
        # [doctring inherited from parent method]

        if unigram:
            raise NotImplementedError("Not required for this homework")

        self.rnn_dim = rnn_dim
        self.e = lexicon.size(1) # dimensionality of word's embeddings
        self.E = lexicon
        self.k = len(tagset)
        self.tag_embeddings = torch.eye(self.k)

        
        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)
        self.init_params()
        


    @override
    def init_params(self) -> None:

        """
            Initialize all the parameters you will need to support a bi-RNN CRF
            This will require you to create parameters for M, M', U_a, U_b, theta_a
            and theta_b. Use xavier uniform initialization for the matrices and 
            normal initialization for the vectors. 
        """
        d = self.rnn_dim
        e = self.e
        k = self.k
        
        # Initialize RNN parameters
        self.M = nn.Parameter(torch.empty(d, 1 + d + e))
        self.M_prime = nn.Parameter(torch.empty(d, 1 +  e + d))

        input_size_A = 1 + d + k + k + d
        input_size_B = 1 + d + k + e + d
        feature_size = d

        self.U_A = nn.Parameter(torch.empty(feature_size, input_size_A))
        self.U_B = nn.Parameter(torch.empty(feature_size, input_size_B))

        self.theta_A = nn.Parameter(torch.empty(feature_size))
        self.theta_B = nn.Parameter(torch.empty(feature_size))

        # Initialize parameters
        nn.init.xavier_uniform_(self.M)
        nn.init.xavier_uniform_(self.M_prime)
        nn.init.xavier_uniform_(self.U_A)
        nn.init.xavier_uniform_(self.U_B)
        nn.init.normal_(self.theta_A)
        nn.init.normal_(self.theta_B)

        
    @override
    def init_optimizer(self, lr: float, weight_decay: float) -> None:
        # [docstring will be inherited from parent]
    
        # Use AdamW optimizer for better training stability
        self.optimizer = torch.optim.AdamW( 
            params=self.parameters(),       
            lr=lr, weight_decay=weight_decay
        )                                   
        self.scheduler = None            
       
    @override
    def updateAB(self) -> None:
        # Nothing to do - self.A and self.B are not used in non-stationary CRFs
        pass

    @override
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Pre-compute the biRNN prefix and suffix contextual features (h and h'
        vectors) at all positions, as defined in the "Parameterization" section
        of the reading handout.  They can then be accessed by A_at() and B_at().
        
        Make sure to call this method from the forward_pass, backward_pass, and
        Viterbi_tagging methods of HiddenMarkovMOdel, so that A_at() and B_at()
        will have correct precomputed values to look at!"""

        n = len(isent)
        d = self.rnn_dim
        e = self.e

        # Initialize h and h_prime lists
        self.h = [torch.zeros(d)]  # h_{-1}
        self.h_prime = [torch.zeros(d)] * (n + 1)  # h'_{n+1}

        # Precompute h_j for j from 0 to n-1
        for j in range(n):
            w_j = isent[j][0]
            w_emb = self.E[w_j]  # Word embedding
            prev_h = self.h[-1]
            input_vec = torch.cat([torch.tensor([1.0]), prev_h, w_emb])  # [1; h_{j-1}; w_j]
            h_j = torch.sigmoid(self.M @ input_vec)
            self.h.append(h_j)

        # Precompute h'_j for j from n to 1
        self.h_prime[n] = torch.zeros(d)  # h'_{n+1}
        for j in range(n - 1, -1, -1):
            w_j_plus_1 = isent[j + 1][0] if j + 1 < n else self.vocab.index(EOS_WORD)
            w_emb = self.E[w_j_plus_1]
            next_h_prime = self.h_prime[j + 1]
            input_vec = torch.cat([torch.tensor([1.0]), w_emb, next_h_prime])  # [1; w_{j+1}; h'_{j+1}]
            h_prime_j = torch.sigmoid(self.M_prime @ input_vec)
            self.h_prime[j] = h_prime_j

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        isent = self._integerize_sentence(sentence, corpus)
        super().accumulate_logprob_gradient(sentence, corpus)

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
        
        """Computes non-stationary k x k transition potential matrix using biRNN 
        contextual features and tag embeddings (one-hot encodings). Output should 
        be ϕA from the "Parameterization" section in the reading handout."""

        k = self.k
        d = self.rnn_dim

        # Retrieve RNN hidden states
        h_i_minus_2 = self.h[position - 2] if position - 2 >= 0 else torch.zeros(d)
        h_prime_i = self.h_prime[position]

        # Prepare tag embeddings (one-hot vectors)
        tag_embeddings = self.tag_embeddings  # Shape: (k, k)

        # Expand and reshape for broadcasting
        h_i_minus_2_exp = h_i_minus_2.view(1, 1, -1).expand(k, k, -1)  # Shape: (k, k, d)
        h_prime_i_exp = h_prime_i.view(1, 1, -1).expand(k, k, -1)      # Shape: (k, k, d)
        s_embeddings_exp = tag_embeddings.unsqueeze(1).expand(k, k, k)  # Shape: (k, k, k)
        t_embeddings_exp = tag_embeddings.unsqueeze(0).expand(k, k, k)  # Shape: (k, k, k)

        # Concatenate inputs
        ones = torch.ones(k, k, 1)  # Shape: (k, k, 1)
        input_tensor = torch.cat(
            [ones, h_i_minus_2_exp, s_embeddings_exp, t_embeddings_exp, h_prime_i_exp],
            dim=2
        )  # Shape: (k, k, input_size)

        # Flatten for batch processing
        input_tensor_flat = input_tensor.view(k * k, -1)  # Shape: (k^2, input_size)

        # Compute f^A
        f_A = torch.sigmoid(input_tensor_flat @ self.U_A.T)  # Shape: (k^2, feature_size)

        # Compute potentials
        potentials_flat = f_A @ self.theta_A  # Shape: (k^2,)

        # Reshape to (k, k)
        potentials = potentials_flat.view(k, k)  # Shape: (k, k)

        return potentials


        
    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
        """Computes non-stationary k x V emission potential matrix using biRNN 
        contextual features, tag embeddings (one-hot encodings), and word embeddings. 
        Output should be ϕB from the "Parameterization" section in the reading handout."""

        k = self.k
        d = self.rnn_dim
        e = self.e

        # Retrieve RNN hidden states and word embedding
        h_i_minus_1 = self.h[position - 1] if position - 1 >= 0 else torch.zeros(d)
        h_prime_i = self.h_prime[position]
        w_i = sentence[position][0]  # Word index at position i
        w_emb = self.E[w_i]          # Shape: (e,)

        # Prepare tag embeddings
        tag_embeddings = self.tag_embeddings  # Shape: (k, k)

        # Expand and reshape for broadcasting
        h_i_minus_1_exp = h_i_minus_1.view(1, -1).expand(k, -1)  # Shape: (k, d)
        h_prime_i_exp = h_prime_i.view(1, -1).expand(k, -1)      # Shape: (k, d)
        w_emb_exp = w_emb.view(1, -1).expand(k, -1)              # Shape: (k, e)

        # Concatenate inputs
        ones = torch.ones(k, 1)  # Shape: (k, 1)
        input_tensor = torch.cat(
            [ones, h_i_minus_1_exp, tag_embeddings, w_emb_exp, h_prime_i_exp],
            dim=1
        )  # Shape: (k, input_size)

        # Compute f^B
        f_B = torch.sigmoid(input_tensor @ self.U_B.T)  # Shape: (k, feature_size)

        # Compute potentials
        potentials = f_B @ self.theta_B  # Shape: (k,)

        # Initialize phi_B
        phi_B = torch.full((k, self.V), float('-inf'))  # Shape: (k, V)

        # Assign potentials to the known word index
        phi_B[:, w_i] = potentials

        return phi_B


