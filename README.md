# movie-sentiment-analysis
Sentiment analysis of movie (IMDB) reviews using dataset provided by the ACL 2011 paper, see http://ai.stanford.edu/~amaas/data/sentiment/.

## Update Session 21st Nov 2018

We so far looked at the data, tried different vectorization techniques (bag of words and TFIDF), trained logistic registration and random forest models.

In this session, we trained a Neural network model with Keras in Sentiment_analysis_of_movies_(IMDB)_with_neural_net.ipynb using a simple bag of words as input.

Next week, we will work on optimizing this neural network with different hyper parameters (nb of layer units, epochs, activation units, layers etc) and plot the results to compare their performances.

In the following weeks, we will
- try different vectorization techniques
- try different NN architectures


### Setup
Use anaconda or virtualenv to create a python 3 environment, then
run `pip install -r requirements.txt' to install the dependencies in your python environment

### Jupyter notebook

`Sentiment_analysis_of_movies_(IMDB).ipynb` created on [Google Colab](https://colab.research.google.com).

### Python source file from the notebook

`sentiment_analysis_of_movies_(imdb).py` extracted from the notebook

### Data 

Can be downloaded separately from http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz, but wont be necessary as the download process has been embedded in the notebook and source file.
