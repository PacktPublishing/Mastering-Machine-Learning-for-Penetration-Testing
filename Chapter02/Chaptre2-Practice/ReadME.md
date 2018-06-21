
# This directory contains the Ling-Spam corpus, as described in the paper:

I. Androutsopoulos, J. Koutsias, K.V. Chandrinos, George Paliouras, 
and C.D. Spyropoulos, "An Evaluation of Naive Bayesian Anti-Spam 
Filtering". In Potamias, G., Moustakis, V. and van Someren, M. (Eds.), 
Proceedings of the Workshop on Machine Learning in the New Information 
Age, 11th European Conference on Machine Learning (ECML 2000), 
Barcelona, Spain, pp. 9-17, 2000.

There are four subdirectories, corresponding to four versions of 
the corpus:

bare: Lemmatiser disabled, stop-list disabled.
lemm: Lemmatiser enabled, stop-list disabled.
lemm_stop: Lemmatiser enabled, stop-list enabled.
stop: Lemmatiser disabled, stop-list enabled.

Each one of these 4 directories contains 10 subdirectories (part1, 
..., part10). These correspond to the 10 partitions of the corpus 
that were used in the 10-fold experiments. In each repetition, one 
part was reserved for testing and the other 9 were used for training. 

Each one of the 10 subdirectories contains both spam and legitimate 
messages, one message in each file. Files whose names have the form
spmsg*.txt are spam messages. All other files are legitimate messages.

By obtaining a copy of this corpus you agree to acknowledge the use 
and origin of the corpus in any published work of yours that makes 
use of the corpus, and to notify the person below about this work.
