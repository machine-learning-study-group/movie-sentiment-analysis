import argparse
import numpy as np

from custom_layers import AverageWords, WordDropout
from preprocess import PreProcessor

from keras.layers import Embedding, Dense, Input, BatchNormalization, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import Adagrad, Adam
from keras import backend as K

embedding_dim = 300
num_hidden_layers = 3
num_hidden_units = 300
num_epochs = 100
batch_size = 100
dropout_rate = 0.2
word_dropout_rate = 0.3
activation = 'relu'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', help='location of dataset', default='data/out_split.pk')
    parser.add_argument('-We', help='location of word embeddings', default='data/glove.6B.300d.txt')
    parser.add_argument('-model', help='model to run: nbow or dan', default='nbow')
    parser.add_argument('-wd', help='use word dropout or not', default='y')

    args = vars(parser.parse_args())

    pp = PreProcessor(args['data'],args['We'])
    pp.tokenize()
    data, labels, data_val, labels_val = pp.make_data()

    embedding_matrix = pp.get_word_embedding_matrix(embedding_dim)

    model = Sequential()

    if args['We'] == "rand":
        model.add(Embedding(len(pp.word_index) + 1,embedding_dim,input_length=pp.MAX_SEQUENCE_LENGTH,trainable=False))
    else:
        model.add(Embedding(len(pp.word_index)+1,embedding_dim,weights=[embedding_matrix],input_length=pp.MAX_SEQUENCE_LENGTH,trainable=False))
    
    if args['wd'] == 'y':
        model.add(WordDropout(word_dropout_rate))
    model.add(AverageWords())

    if args['model'] == 'dan':
        for i in range(num_hidden_layers):
            model.add(Dense(num_hidden_units))
            model.add(BatchNormalization())
            model.add(Activation(activation))
            model.add(Dropout(dropout_rate))

    model.add(Dense(labels.shape[1]))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Activation('softmax'))

    adam = Adam()
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['categorical_accuracy'])

    model.summary()

    # get_embedding_layer_output = K.function([model.layers[0].input],[model.layers[0].output])
    # el_output = np.mean(get_embedding_layer_output([data])[0],axis=1)
    # print el_output

    # get_average_word_layer_output = K.function([model.layers[0].input],[model.layers[1].output])
    # print get_average_word_layer_output([data])[0]

    model.fit(data,labels,batch_size=batch_size,epochs=num_epochs,validation_data=(data_val,labels_val))

    
