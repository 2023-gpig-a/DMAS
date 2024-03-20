# DMAS - Data Management and Analysis System

This system is responsible for the following:
* Ingesting raw image data into the database
* Analysing raw image data to extract plant data
* Matching similar plants and assigning a `plant_id` to them
* Tracking the growth of plants over time

[//]: # (TODO update this to include building the models)
## Running without docker
```
python3 -m venv venv
. venv/bin/activate (linux) or ./venv/Scripts/activate (win)
python3 -m pip install -e
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Running with docker
```
docker build -t dmas_image .
docker run -d --name dmas -p 8080:8080 dmas_image
```

## Endpoints:

`/upload_image`
Post images to this endpoint, if it does not contain humans we create PostgreSQL entries for it, storing the longitude, latitude, file location, and a timestamp.
We will extract this data from the image metadata.
We then process these images to identify the plants contained.

Returns: `{status: StatusEnum, message: str}`

`/track_growth`
When a get request is made to this endpoint, the DMAS will look at all of our processed data, arrange it based on date, and return a key value map of `plant_id` to plant growth data.

Returns: `{plant_id: plant_growth_data}`