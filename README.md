# movie-sentiment-analysis
Sentiment analysis of movie (IMDB) reviews using dataset provided by the ACL 2011 paper, see http://ai.stanford.edu/~amaas/data/sentiment/.

## Update Session 23rd Jan 2019 

**We so far**
- explored the data in 1_data_exploration.ipynb
- tried different vectorization techniques (bag of words and TFIDF) in 2_spot_check_algos.ipynb
- check multiple algorithms and compared performances: logistic regression, random forest and NN models. 
- started optimizing our NN model (work on dropout) in 3_opti_neural_net.ipynb
- tried different architectures in experiments folder: LSTM, DAN, CNN

**In this session**, 

We completed the implementation of a CNN in Keras with multiple channels, in experiments/keras_cnn_with_multiple_channels.ipynb, using the following:

http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/ sentence classification with CNN with tensorflow
https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/ cnn in keras with multiple channels

We implemented the multiple channels and trained on our data, reaching around 80% acc on test data. We started adding pre trained embedding from our work on dan cnn in experiments/keras_cnn_with_pretrained_embedding.ipynb

**Next session 30th Jan**, 
recommended homework: please read before coming the article from wildml above and check our work so far in experiments/keras_cnn_with_multiple_channels.ipynb and experiments/keras_cnn_with_pretrained_embedding.ipynb (WIP). If you don't, it's going to be really hard to follow ;).

We will carry on the work of the previous session and fix the CNN with the pre trained embedding from Glove. We will compare the results of training the embedding on our dataset vs not doing it.

If we have enough time, We will define a reasonnable dataset and vocabulary size and train with those parameters, then compare the accuracies obtained. We will run the training on GPU on colab and compare the amounts of time to process it.

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
