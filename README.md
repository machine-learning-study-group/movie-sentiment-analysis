# movie-sentiment-analysis
Sentiment analysis of movie (IMDB) reviews using dataset provided by the ACL 2011 paper, see http://ai.stanford.edu/~amaas/data/sentiment/.

## Update Session 19th Dec 2018 

**We so far**
- explored the data in 1_data_exploration.ipynb
- tried different vectorization techniques (bag of words and TFIDF) in 2_spot_check_algos.ipynb
- check multiple algorithms and compared performances: logistic registration, random forest and NN models. 
- started optimizing our NN model (work on dropout) in 3_opti_neural_net.ipynb
- tried different architectures in experiments folder: LSTM, DAN

**In this session**, 

we read that research paper on a simple deep NN competing with more complex NN (like CNN and RNN) on sentiment analysis: 
Deep Averaging Network DAN
https://cs.umd.edu/~miyyer/pubs/2015_acl_dan.pdf

implemented on experiments/keras_dan.ipynb:

**Next session 9th Jan**, 

As a prerequisite, please get familiar with that research paper on sentence classification with CNN, and related article (tensorflow rewrite): 
https://arxiv.org/abs/1408.5882
http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

We will implement a Convolutional Neural Network following those articles

one of our members committed an implementation of CNN with keras in experiments/keras_nn_cnn.ipynb, you can also have a look at it


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
