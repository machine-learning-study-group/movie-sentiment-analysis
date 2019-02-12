# movie-sentiment-analysis
Sentiment analysis of movie (IMDB) reviews using dataset provided by the ACL 2011 paper, see http://ai.stanford.edu/~amaas/data/sentiment/.

## Update Session 6th Feb 2019 

**We so far**
- explored the data in 1_data_exploration.ipynb
- tried different vectorization techniques (bag of words and TFIDF) in 2_spot_check_algos.ipynb
- check multiple algorithms and compared performances: logistic regression, random forest and NN models. 
- started optimizing our NN model (work on dropout) in 3_opti_neural_net.ipynb
- tried different architectures in experiments folder: LSTM, DAN, CNN
- tried with pre trained embeddings from Glove and Google (word2vec)

We have so far trained many different models from ML (logistic reg, random forest) to DL (simple NN, CNN, RNN) with multiple vectorization techniques and with the use in some case of pre trained embeddings. 
We are going to go through each one of them again, format them nicely, train them on the whole dataset on google colab using GPU and on as many epochs as possible until it reaches a plateau on the accuracy of the validation data set. We will bring together the results of each training and plot the performances and time to run to compare the different models, vectorization techniques and use or not of pre trained embedding.

**In this session**, 
(check the PRs)
Mani started working the simple NN models
Dave/Michael on dan network
Jeremie/Randall on ML models (random forest): need rework since can't run on whole dataset with current vectorization technique

**Next session 13th Feb**, 
no homework for this session

will carry on with previous work 

**In the following weeks**, we will
- train again all previously used models on whole dataset and try to get best results out of each model
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
