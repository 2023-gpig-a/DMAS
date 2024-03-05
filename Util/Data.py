from datetime import datetime
from pydantic import BaseModel
from enum import Enum


# Each species listed here requires a model that can detect them from a series of images
class IdentifiedSpecies(Enum):
    Unidentified = 0
    JapaneseKnotweed = 1


class Image(BaseModel):
    latitude: float
    longitude: float
    # image: TODO Not sure how we send this over the wire


# Database entries start
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
# Database entries end
