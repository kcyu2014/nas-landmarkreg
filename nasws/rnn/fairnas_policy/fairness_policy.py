"""
FairNAS of training:
Reference: https://arxiv.org/pdf/1907.01845.pdf
Chu et al. Xiaomi AI Lab, arxiv.org, 2019.


This is a new way of training NAS, instead of training the superNet at once,
it trains the each layer (node) in superNet one by one,
For each layer in SuperNet:
    for m in range(NumPossibleConnection of each layer)

        grad


July 4th.
I ask the code, but maybe impossible to get this code. Reimplement as a interesting baseline or
just use as part of our research.

July 12th.
Update: the origin
"""
