FROM python:3.8.7-buster

RUN apt-get update &&\
    apt-get install -y binutils libproj-dev gdal-bin libgdal-dev

# These two environment variables prevent __pycache__/ files.
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt
RUN pip install GDAL==2.4.0 --global-option=build_ext --global-option="-I/usr/include/gdal/"

ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH