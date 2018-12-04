# movie-sentiment-analysis
Sentiment analysis of movie (IMDB) reviews using dataset provided by the ACL 2011 paper, see http://ai.stanford.edu/~amaas/data/sentiment/.

## Update Session 28th Nov 2018

We so far looked at the data, tried different vectorization techniques (bag of words and TFIDF), trained logistic registration, random forest and NN models.

In this session, 
- We separated our data into training and validation sets (we would use the validation set later to evaluate the accuracy of our algorithm once we tested it on multiple models, with different hyper parameters).
- We splitted our training set into 3 folds training + cross validation set. So that we test 3 times a model with 3 different splits of our data, and get the mean accuracy in the end.
- Ran it on Keras NN, and plotted the results on accuracy across the 3 folds.

- We also quickly experimented google colab at the very end. We'll try to use it from next session to get everyone to participate.

Next week, we will work on optimizing this neural network with different hyper parameters (nb of layer units, epochs, activation units, layers etc) and plot the results to compare their performances.

In the following weeks, we will
- try different vectorization techniques
- try different NN architectures


### Setup
Use anaconda or virtualenv to create a python 3 environment, then
run `pip install -r requirements.txt' to install the dependencies in your python environment

### Jupyter notebook
we have multiple notebooks, check the commits to see which one we worked on in last session. Check also the pull requests in case they haven't been merged yet.

`Sentiment_analysis_of_movies_(IMDB).ipynb` created on [Google Colab](https://colab.research.google.com).


### Data 

Can be downloaded separately from http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz, but wont be necessary as the download process has been embedded in the notebook and source file.
