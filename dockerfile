FROM python:3.8
MAINTAINER Lilu "chenglilu1027@gmail.com"
#COPY ./* ./app/
COPY ./train.py ./app/train.py
COPY ./app.py ./app/app.py
COPY ./inference.py ./app/inference.py
COPY ./gender.h5 ./app/gender.h5
COPY ./requirements.txt ./app/requirements.txt

WORKDIR /app/
RUN pip3 install -r requirements.txt
#EXPOSE 5000
CMD ["python3", "app.py"]