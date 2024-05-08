# DMAS - Data Management and Analysis System

This system is responsible for the following:
* Ingesting raw image data into the database
* Analysing raw image data to extract plant data
* Matching similar plants and assigning a `plant_id` to them
* Tracking the growth of plants over time

## Setup
### Configuration files
```
mv config.ini.example config.ini
```
Fill in config.ini with the username and password to your postgresql database.
Paste your plantnet API key into the api_key section, this can be created at https://my.plantnet.org/.

### Python venv
```
python3 -m venv venv
. venv/bin/activate (linux) or ./venv/Scripts/activate (win)
python3 -m pip install -e
pip install -r requirements.txt
```

### Train the human detection model
```
python3 models/human_detection/human_detector.py
```

## Running without docker
```
uvicorn app.main:app --reload
```

## Running with docker
```
docker build -t dmas_image .
docker run -d --name dmas -p 8080:8080 dmas_image
```

## Endpoints:

You can view the endpoints and accompanying API doc by running the service, then going to http://hostname:port/docs.
