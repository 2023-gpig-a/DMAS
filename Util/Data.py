from datetime import datetime
from pydantic import BaseModel
from Util.IdentifiedSpecies import IdentifiedSpecies


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
