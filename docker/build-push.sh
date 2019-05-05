#! /bin/bash
docker image build -t ml-group .
docker tag ml-group davesnowdon/ml-study-group-movie-sentiment
docker push davesnowdon/ml-study-group-movie-sentiment
