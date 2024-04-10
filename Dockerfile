FROM python:3.8-slim-buster

WORKDIR /portfolio-predictor

COPY ./app ./app
COPY ./models ./models

RUN apt-get update && apt-get install -y curl build-essential

RUN /bin/bash ./app/install-ta-lib.sh
RUN pip install -r ./app/requirements.txt
EXPOSE 8000
CMD cd app && gunicorn app:app

