FROM python:3.9

RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get -y install libpq-dev gcc \
    && pip install psycopg2

# install requirements before adding app
COPY ./requirements.txt /tmp
RUN pip install \
    --disable-pip-version-check \
    --no-python-version-warning \
    --no-cache-dir --upgrade -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

# expose default non-privileged HTTP port
EXPOSE 8080/tcp

COPY . /app
WORKDIR /app

# make sure the weights have been copied in
# fails if they haven't
RUN [ -f "./models/human_detection/weights/human_classification_results.pkl" ]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
