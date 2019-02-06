# movie-sentiment-analysis
Sentiment analysis of movie (IMDB) reviews using dataset provided by the ACL 2011 paper, see http://ai.stanford.edu/~amaas/data/sentiment/.

## Update Session 30th Jan 2019 

**We so far**
- explored the data in 1_data_exploration.ipynb
- tried different vectorization techniques (bag of words and TFIDF) in 2_spot_check_algos.ipynb
- check multiple algorithms and compared performances: logistic regression, random forest and NN models. 
- started optimizing our NN model (work on dropout) in 3_opti_neural_net.ipynb
- tried different architectures in experiments folder: LSTM, DAN, CNN
- tried with pre trained embeddings from Glove and Google (word2vec)

**In this session**, 

We completed the implementation of a CNN in Keras with multiple channels and pre trained embedding, in experiments/keras_cnn_with_pretrained_embedding.ipynb, using the following:

http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/ sentence classification with CNN with tensorflow
https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/ cnn in keras with multiple channels

We ran the two cnn notebooks with and without pre trained embedding on Google Colab with gpu (way faster than on cpu on our laptops), on the whole imdb dataset (12.5k records instead of 3k) and reached around 89% acc on test data with pre trained embedding. 

**Next session 6th Feb**, 
no homework for this session

We have so far trained many different models from ML (logistic reg, random forest) to DL (simple NN, CNN, RNN) with multiple vectorization techniques and with the use in some case of pre trained embeddings. We are going to go through each one of them again, format them nicely, train them on the whole dataset on google colab and on as many epochs as possible until it reaches a plateau on the accuracy of the validation data set. We will bring together the results of each training and plot the performances and time to run to compare the different models, vectorization techniques and use or not of pre trained embedding.


**In the following weeks**, we will
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
