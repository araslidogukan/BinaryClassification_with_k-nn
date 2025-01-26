## K NEAREST NEIGHBOURS AND VARIANTS FOR BINARY CLASSIFICATION

This repository includes k-nn, modified k-nn, fuzzy k-nn with discrete and continuous class membership values, r near neighbours and r near neighbours with modified k-nn approach implementations from scratch to perform binary classification on Python using numpy.

Additionally, partition.py includes implementations of random train-test splitting (with each permutation having equal probability) and folding algorithm for cross-validation. MinMaxScaler.py includes min-max scaling as k-nn requires attributes to be of the same scale.

"main.py" shows how to compare these algorithms both with single train-test split and 5 _times_ 5 folding.
