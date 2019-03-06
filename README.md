# movie-sentiment-analysis
Sentiment analysis of movie (IMDB) reviews using dataset provided by the ACL 2011 paper, see http://ai.stanford.edu/~amaas/data/sentiment/.

## Update Session 6th March 2019 

**We so far**
- explored the data in 1_data_exploration.ipynb
- tried different vectorization techniques (bag of words and TFIDF) in 2_spot_check_algos.ipynb
- check multiple algorithms and compared performances: logistic regression, random forest and NN models. 
- started optimizing our NN model (work on dropout) in 3_opti_neural_net.ipynb
- tried different architectures in experiments folder: LSTM, DAN, CNN
- tried with pre trained embeddings from Glove and Google (word2vec)
- train again all previously used models on whole dataset and try to get best results out of each model
- plot the results of all different architectures and compare them

We have so far trained many different models from ML (logistic reg, random forest) to DL (simple NN, CNN, RNN) with multiple vectorization techniques and with the use in some case of pre trained embeddings. 
We are going to go through each one of them again, format them nicely, train them on the whole dataset on google colab using GPU and on as many epochs as possible until it reaches a plateau on the accuracy of the validation data set. We will bring together the results of each training and plot the performances and time to run to compare the different models, vectorization techniques and use or not of pre trained embedding.

**In this session**, 
very short one (everyone stuck at work turned up late)
review and small improvements on ML notebook. discussed next steps for the study group.

**Next session 13th March**, 
We will carry on with our project on sentiment analysis. 
We will train the NN models on colab again on a larger dataset (50k instead of 12.5k), use the same metric everywhere (f1 score), add the time it took to train the model.
Then we will compare the results again, compare them to those of the research papers we worked on, and try to interpret them.


**In the following weeks**, we will
- interpret the results, standardize the results
- move to another project


### Setup
#### Google Colab
Import notebooks in google colab. (very recommended if you are working on neural networks because Colab provides GPU)
You will need to uncomment some lines in your notebook to build the environment.

#### Local Installation
we use Jupyter Lab, or Jupyter classic notebook.
Use anaconda or virtualenv to create a python 3 environment, then
run `pip install -r requirements.txt' to install the dependencies in your python environment

### Jupyter notebook
we have multiple notebooks, check the commits to see which one we worked on in last session. Check also the pull requests in case they haven't been merged yet.


### Data 

Can be downloaded separately from http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz, but wont be necessary as the download process has been embedded in the notebook and source file.
