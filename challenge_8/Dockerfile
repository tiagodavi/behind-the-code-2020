FROM python:3.8-alpine

WORKDIR /opt/app

COPY . /opt/app

EXPOSE 8000

RUN pip install --upgrade pip flask gunicorn flask-cors ibm-watson watson-developer-cloud six