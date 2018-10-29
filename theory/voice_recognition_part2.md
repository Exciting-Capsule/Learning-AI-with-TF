After completing part 1, the next step is to insert a simple patern recognition to recognice the variant of sample date. In theoritical, saying "Hello" in different pronounciation and tone should have similar wavelength pattern. based on the sample data available, the patern recognition feature should be able to identify the same word that have different tone and pronounciation. 

For experiment, use one of the basic patern recognition algorithm as below:

1. k-nearest neighbors algorithm. (reference: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
2. Cosine similarity. (reference: https://en.wikipedia.org/wiki/Cosine_similarity, R code reference: https://stackoverflow.com/questions/2535234/find-cosine-similarity-between-two-arrays)

For advance user, adapting deep learning will be a better solution, one of the simple approach which already available in TensorFlow.js:
1. Neural network. (TensorFlow.js reference: https://js.tensorflow.org/tutorials/webcam-transfer-learning.html)
