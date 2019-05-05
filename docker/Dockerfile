# Docker tensorflow docs https://hub.docker.com/r/tensorflow/tensorflow/

# Build as follows:
# docker image build -t ml-group .

# Run this as follows to mount current directory for notebooks
# docker container run -it --rm -u $(id -u):$(id -g) -p 8888:8888 -v $(pwd):/home/jupyter/notebooks ml-group

# or if using image from docker hub
# docker container run -it --rm -u $(id -u):$(id -g) -p 8888:8888 -v $(pwd):/home/jupyter/notebooks davesnowdon/ml-study-group-movie-sentiment

# uncomment this to use GPU
#FROM tensorflow/tensorflow:latest-gpu-py3 python
# uncomment this to use CPU
FROM tensorflow/tensorflow:latest-py3

# NOTE: This appears to require CUDA 10
# if you want to use a GPU, you'll need the NVIDIA runtime.
# See https://devblogs.nvidia.com/gpu-containers-runtime/
# On ubuntu run the following:
# curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
# sudo apt-get update
# sudo apt-get install -y nvidia-docker2
# sudo pkill -SIGHUP dockerd
# verify all looks good by running
# sudo nvidia-container-cli --load-kmods info

RUN pip install wget pandas matplotlib nltk keras scikit-learn jupyterlab nbdime

# enable nbdime Jupyter notebook diffing extension
RUN nbdime extensions --enable --system

RUN mkdir /home/jupyter
RUN chmod 777 /home/jupyter
RUN mkdir /home/jupyter/notebooks
RUN chmod 777 /home/jupyter/notebooks
WORKDIR /home/jupyter
RUN HOME=/home/jupyter python3 -c "import nltk ; nltk.download('punkt') ; nltk.download('stopwords')"

CMD HOME=/home/jupyter JUPYTER_ENABLE_LAB=yes jupyter lab --ip=0.0.0.0