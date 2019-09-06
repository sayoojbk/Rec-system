FROM python:3

COPY . /recommendation

WORKDIR /recommendation

RUN ls

RUN apt-get install gcc

RUN pip install -r requirements.txt  

CMD ["python", "manage.py", "run"]

