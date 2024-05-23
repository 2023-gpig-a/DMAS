from datetime import datetime
import os
from pydantic import BaseModel
from configparser import ConfigParser
import psycopg2
import numpy as np
from sklearn import linear_model


# Database entries start
class RawEntry(BaseModel):
    latitude: float
    longitude: float
    image_uri: str
    date: datetime


class ProcessedEntry(BaseModel):
    image_uri: str
    plant_id: str
# Database entries end


def load_plant_net_api_key(filename=os.getenv('CONFIG_FILE', "config.ini")):
    parser = ConfigParser()
    parser.read(filename)

    # get section
    config = {}
    if parser.has_section('plantnet'):
        params = parser.items('plantnet')
        for param in params:
            config[param[0]] = param[1]
    else:
        raise Exception(f'Section plantnet not found in the {filename} file')

    return config


def load_database_config(filename=os.getenv('CONFIG_FILE', "config.ini")):
    parser = ConfigParser()
    parser.read(filename)

    # get section
    config = {}
    if parser.has_section('postgresql'):
        params = parser.items('postgresql')
        for param in params:
            config[param[0]] = param[1]
    else:
        raise Exception(f'Section postgresql not found in the {filename} file')

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


def get_species_data(conn, center_lat: float = 0.0, center_lon: float = 0.0, scan_range: float = 360, day_range: int = 1000):
    SQL = f"""
    SELECT 
        plant_id, 
        ARRAY_AGG(
            ARRAY[
                TO_CHAR(date, 'YYYY-MM-DD'),  
                latitude::text,
                longitude::text,
                count::text
            ] ORDER BY date
        ) AS date_counts
    FROM (
        SELECT 
            t1.plant_id, 
            t2.date, 
            t2.latitude,
            t2.longitude,
            COUNT(*) AS count
        FROM 
            image_processing.processed_entry t1
        JOIN 
            image_processing.raw_entry t2 ON t1.image_uri = t2.image_uri
        WHERE
            (({center_lat} - t2.latitude)^2 + ({center_lon} - t2.longitude)^2) <= {scan_range}^2
            AND t2.date BETWEEN CURRENT_DATE - {day_range} AND CURRENT_DATE 
            AND (t2.seen OR t2.date < (NOW() - INTERVAL '2 days'))
        GROUP BY 
            t1.plant_id, t2.date, t2.latitude, t2.longitude
    ) AS subquery
    GROUP BY 
        plant_id;
    """

    with conn.cursor() as cursor:
        cursor.execute(SQL)
        data = cursor.fetchall()
        return data


def get_counts_for_species_by_date(conn):

    def count_dates(datum):
        out = {}
        for date, _, _, count in datum:
            if date not in out:
                out[date] = int(count)
            else:
                out[date] += int(count)
        return out

    species_data = get_species_data(conn)
    species_counts = {s: count_dates(d) for s, d in species_data}
    return species_counts


def get_species_daily_growth(conn):
    # For each species a map of date d:count scanned on date d
    species_counts = get_counts_for_species_by_date(conn)

    # For each species perform RANSAC and get gradient of growth g
    species_gradients = {}
    for species, c in species_counts.items():
        # Extract dates and counts as individual numpy arrays
        count_data = np.array(list(c.items()))
        dates, counts = count_data[:, 0], count_data[:, 1].astype(int)

        # Convert strings to datetime objects and calculate the offset in days from first date
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
        days_difference = [int((date - dates[0]).days) for date in dates]
        days_difference_array = np.array(days_difference)

        # Fit dates and counts to RANSAC and linear model, and extract gradient
        ransac = linear_model.RANSACRegressor()
        ransac.fit(days_difference_array.reshape(-1, 1), counts.reshape(-1, 1))
        gradient = ransac.estimator_.coef_[0][0]
        gradient = 0 if gradient is None else gradient  # ransac coef is None if flat
        species_gradients[species] = gradient

    return species_gradients
