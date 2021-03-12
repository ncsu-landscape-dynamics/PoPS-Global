FROM python:3.8.7-buster

RUN apt-get update &&\
    apt-get install -y binutils libproj-dev gdal-bin libgdal-dev

# These two environment variables prevent __pycache__/ files.
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# RUN apt-get install -y software-properties-common && apt-get update
# RUN apt-get install --reinstall build-essential
# RUN apt-get install -y python3.7-dev
# RUN apt-get install -y python-rtree gdal-bin libgdal-dev
# ARG CPLUS_INCLUDE_PATH=/usr/include/gdal
# ARG C_INCLUDE_PATH=/usr/include/gdal
# apt-get install -y build-essential binutils libproj-dev python-rtree gdal-bin libgdal-dev python-dev

# ADD . /app
WORKDIR /app
COPY . /app
# ADD ./requirements.txt /app/requirements.txt
# COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt
RUN pip install GDAL==2.4.0 --global-option=build_ext --global-option="-I/usr/include/gdal/"

ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH