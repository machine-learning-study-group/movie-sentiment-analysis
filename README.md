# movie-sentiment-analysis
Sentiment analysis of movie (IMDB) reviews using dataset provided by the ACL 2011 paper, see http://ai.stanford.edu/~amaas/data/sentiment/.

## Update Session 12th Dec 2018

**We so far **
- explored the data in 1_data_exploration.ipynb
- tried different vectorization techniques (bag of words and TFIDF) in 2_spot_check_algos.ipynb
- check multiple algorithms and compared performances: logistic registration, random forest and NN models. 
- started optimizing our NN model (work on dropout) in 3_opti_neural_net.ipynb

**In this session**, 

we read those articles:
- https://github.com/tensorflow/workshops/blob/master/extras/keras-bag-of-words/keras-bow-model.ipynb to understand the notions of bag of words, tokenizer, sparse matrix, etc
- https://towardsdatascience.com/a-beginners-guide-on-sentiment-analysis-with-rnn-9e100627c02e to work on RNN

work on contributions/keras_lstm_nn.ipynb:
- we came back to our work on LSTM and fixed it: we were using previously the wrong vectorization technique to work with RNN: for each row we needed to use a vector of indices (which keeps the order of words) rather than a sparse vector (from bag of words).

**Next week**, 
we will read the chapter 1 (Text Classification) and its mentioned articles  of
https://machinelearningmastery.com/applications-of-deep-learning-for-natural-language-processing/

and work to implement a Convolutional Neural Network

**In the following weeks**, we will
- try different NN architectures
- plot the results of all different architectures and compare them
- optimize our NN model
- try cloud based APIs to compare how they perform against our models


### Setup
Use anaconda or virtualenv to create a python 3 environment, then
run `pip install -r requirements.txt' to install the dependencies in your python environment

### Jupyter notebook
we have multiple notebooks, check the commits to see which one we worked on in last session. Check also the pull requests in case they haven't been merged yet.


### Data 

Can be downloaded separately from http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz, but wont be necessary as the download process has been embedded in the notebook and source file.
