The idea behind the project is to implement the PyTorch API using only NumPy/CuPy for vectorization. The goal is to achieve performance comparable to PyTorch. 

At this stage, much is not implemented. I want to implement the basic features first so that I can run the training pipeline from scratch.

A major omission at the moment is the absence of "device". The current workaround is to automatically choose between CuPy/Numpy depending on whether a GPU is present.
