from datetime import datetime
from typing import List
from pydantic import BaseModel
from configparser import ConfigParser
import psycopg2


# Database entries start
class RawEntry(BaseModel):
    latitude: List[float]
    longitude: List[float]
    image_uri: str
    date: datetime


class ProcessedEntry(BaseModel):
    image_uri: str
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


def insert_raw_entry(conn, entry: RawEntry) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """ 
            INSERT INTO image_processing.raw_entry (image_uri, latitude, longitude, date) 
            VALUES (%s, %s, %s, %s); 
            """, (entry.image_uri,entry.latitude,entry.longitude,entry.date,))
    conn.commit()


def insert_processed_entry(conn, entry: ProcessedEntry) -> None:
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO image_processing.processed_entry (image_uri, plant_id) 
            VALUES (%s, %s); 
        """, (entry.image_uri, entry.plant_id))
    conn.commit()
