FROM python:3.9

# install requirements before adding app
COPY ./requirements.txt /tmp
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

# expose default non-privileged HTTP port
EXPOSE 8080/tcp

COPY . /app
WORKDIR /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]