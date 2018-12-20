from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# import cPickle as pkl
import _pickle as pkl
import numpy as np

class PreProcessor:
    def __init__(self,TRAIN_DATA,VAL_DATA,WE_FILE):
        self.reviews = TRAIN_DATA
        self.reviews_val = VAL_DATA
        self.we_file = WE_FILE

    def tokenize(self):
        print(self.reviews[0])

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.reviews)

        self.sequences = tokenizer.texts_to_sequences(self.reviews)
        self.sequences_val = tokenizer.texts_to_sequences(self.reviews_val)

        self.word_index = tokenizer.word_index
        print("Found %s unique tokens" %(len(self.word_index)))

    def get_word_embedding_matrix(self,EMBEDDING_DIM=100):
        embeddings_index = {}

        if self.we_file == "rand":
            return None

        f = open(self.we_file)

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))

        self.embedding_matrix = np.zeros((len(self.word_index)+1, EMBEDDING_DIM))

        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

        return self.embedding_matrix

if __name__ == "__main__":
	pass
