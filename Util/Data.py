from datetime import datetime
from pydantic import BaseModel
from enum import Enum
from configparser import ConfigParser
import psycopg2


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
    image_uri: str
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


def load_config(section='postgresql', filename='database.ini'):
    parser = ConfigParser()
    parser.read(filename)

    # get section
    config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            config[param[0]] = param[1]
    else:
        raise Exception(f'Section {section} not found in the {filename} file')

    return config


def connect(config):
    """ Connect to the PostgreSQL database server """
    try:
        # connecting to the PostgreSQL server
        with psycopg2.connect(**config) as conn:
            print('Connected to the PostgreSQL server.')
            return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)


def insert_raw_entry(cursor, entry: RawEntry) -> None:
    print(entry)
    cursor.execute(f"""
        INSERT INTO image_processing.raw_entry (image_uri, latitude, longitude, date) 
        VALUES ({entry.image_uri}, {entry.latitude} , {entry.longitude}, {entry.date}); 
    """)