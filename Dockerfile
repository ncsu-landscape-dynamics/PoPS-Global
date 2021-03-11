FROM python:3.7-buster

RUN apt-get update
# RUN apt-get install -y software-properties-common && apt-get update
RUN apt-get install --reinstall build-essential
RUN apt-get install -y python3.7-dev
RUN apt-get install -y python-rtree gdal-bin libgdal-dev
ARG CPLUS_INCLUDE_PATH=/usr/include/gdal
ARG C_INCLUDE_PATH=/usr/include/gdal
# apt-get install -y build-essential binutils libproj-dev python-rtree gdal-bin libgdal-dev python-dev

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app
# COPY requirements.txt /app/requirements.txt

RUN python -m pip install -r requirements.txt

