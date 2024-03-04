from datetime import datetime
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from Util.IdentifiedSpecies import IdentifiedSpecies

app = FastAPI()


class Image(BaseModel):
    latitude: float
    longitude: float
    # image: TODO Not sure how we send this over the wire


class RawEntry(BaseModel):
    latitude: float
    longitude: float
    raw_entry: str
    date: datetime


class ProcessedEntry(BaseModel):
    latitude: float
    longitude: float
    plant_id: str
    date: datetime


class PlantIDMapEntry(BaseModel):
    species: IdentifiedSpecies
    plant_id: str


@app.get("/")
async def hello_world():
    return {"response": "Hello World"}


@app.post("/upload_images")
async def upload_images(image: Image):
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED
    )

    # If the image contains a human exit
    # First upload image to s3 storage, this returns uri
    # Create raw entry and upload to postgresql
    # Process the raw entry and upload the processed entry to postgresql
    # If we have created a new plant id then run it against our plant id mapper
    #   If we have found a mapping, create PlantIDMapEntry and upload to postgresql


async def process_raw_images(raw_entry: RawEntry):
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED
    )

    # Find all plants within the image
    # Cluster and identify these
    # Return processed entry


@app.get("/track_growth")
async def track_growth():
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED
    )
