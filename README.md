# movie-sentiment-analysis
Sentiment analysis of movie (IMDB) reviews using dataset provided by the ACL 2011 paper, see http://ai.stanford.edu/~amaas/data/sentiment/.

## Update Session 9th Jan 2019 

**We so far**
- explored the data in 1_data_exploration.ipynb
- tried different vectorization techniques (bag of words and TFIDF) in 2_spot_check_algos.ipynb
- check multiple algorithms and compared performances: logistic registration, random forest and NN models. 
- started optimizing our NN model (work on dropout) in 3_opti_neural_net.ipynb
- tried different architectures in experiments folder: LSTM, DAN

**In this session**, 

we read that research paper on sentence classification with CNN, and two related article (tensorflow rewrite, explanation of CNN use in NLP): 
https://arxiv.org/abs/1408.5882
http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/

We went through the code of the following files of the tensorflow implementation https://github.com/dennybritz/cnn-text-classification-tf to understand it: train.py, eval.py and text_cnn.py

**Next session 16th Jan**, 

We will carry on the work of the previous session and reimplement it in Keras (it should be way simpler than in tensorflow).

We will add if we have enough time, the pre trained embedding with word2vec.

As a prerequesite, please read at least the 2 articles from wildml.

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
