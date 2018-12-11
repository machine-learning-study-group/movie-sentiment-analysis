# movie-sentiment-analysis
Sentiment analysis of movie (IMDB) reviews using dataset provided by the ACL 2011 paper, see http://ai.stanford.edu/~amaas/data/sentiment/.

## Update Session 5th Dec 2018

We so far 
- explored the data in 1_data_exploration.ipynb
- tried different vectorization techniques (bag of words and TFIDF) in 2_spot_check_algos.ipynb
- check multiple algorithms and compared performances: logistic registration, random forest and NN models. 
- started optimizing our NN model (work on dropout) in 3_opti_neural_net.ipynb

In this session, 
work on 3_opti_neural_net.ipynb:
- Optimize our NN on trying no / low / high dropout on 1 / 2 layers
- try another NN architecture using embedding layer
- used google colab during the session

Next week, we will carry on with optimization, try different hyper parameters (nb of layer units, epochs, activation units, layers etc) and plot the results to compare their performances.

In the following weeks, we will
- try different vectorization techniques
- optimize our NN model
- try different NN architectures


### Setup
Use anaconda or virtualenv to create a python 3 environment, then
run `pip install -r requirements.txt' to install the dependencies in your python environment

### Jupyter notebook
we have multiple notebooks, check the commits to see which one we worked on in last session. Check also the pull requests in case they haven't been merged yet.


### Data 

Can be downloaded separately from http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz, but wont be necessary as the download process has been embedded in the notebook and source file.
