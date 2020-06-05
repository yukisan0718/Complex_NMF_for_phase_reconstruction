Complex NMF for phase reconstruction
====

## Overview
Implementation of complex NMF algorithms proposed by H. Kameoka et. al. [1-2].

The conventional NMF implicitly assumes additivity of magnitude (or power) spectrum, and does not take account of phase information. On the contrary, the "complex_NMF.py" is based on an algorithm using the phase information, it can reconstruct the original phase spectrum.


## Requirement
matplotlib 3.1.0

numpy 1.18.1

scipy 1.4.1

museval 0.3.0 (only for evaluation metrics)


## Dataset preparation
You can apply this algotithm to any application you want. An example of a short piano play has been prepared for the demonstration.


## References
[1] H. Kameoka, N. Ono, K. Kashino, and S. Sagayama: 'Complex NMF: A New Sparse Representation for Acoustic Signals', in Proceedings of International Conference on Acoustics, Speech, and Signal Processing (ICASSP), pp.3437–3440, (2009)

[2] H. Kameoka, H. Kagami, and M. Yukawa: 'Complex NMF with the Generalized Kullback-Leibler Divergence', in Proceedings of International Conference on Acoustics, Speech, and Signal Processing (ICASSP), pp.56–60, (2017)