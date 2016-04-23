# GANNET
The Go Artifical Neural Network Library

## About
GANNET is a simple Go library implementing basic backpropagation algorithm for
neural networks learning. Since the learning algorithm parameters are not very
configurable yet and the only prepared acivation function is the standard
sigmoid it is really not ready for any real-world usage.

There are also several Go-specific features that need to be redesigned: the
activation functions management and parallelization with goroutines are
particulary silly.
