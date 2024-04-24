import random
import uuid
import math
from datetime import datetime, timedelta

from util.data import RawEntry, ProcessedEntry, insert_raw_entry, insert_processed_entry, load_database_config, connect


def growth_map(
        x: float,
        max_count: int = 20,
        min_count: int = 15,
        offset: int = 0,
        scale_factor: float = 30,
        growth_factor: float = 0):
    # Generate sinusoidal data for the year spread from min-max count
    # Offset allows us to simulate different species having different growth patterns

    x_scaled = x/scale_factor + offset      # Stretch the graph over a year and apply offset
    out = (math.sin(x_scaled) + 1)/2        # range [0,1]
    out *= (max_count - min_count)          # range [0, max_count-min_count]
    out += min_count                        # range [min_count, max_count]
    out += growth_factor * x                # Simulate the species growing or shrinking over time
    return max(0, out)


def generate_entry(conn, date: datetime, plant_id: str) -> None:
    image_uri = uuid.uuid4().hex

    raw_entry = RawEntry(
        latitude=random.random(),
        longitude=random.random(),
        image_uri=image_uri,
        date=date
    )
    insert_raw_entry(conn, raw_entry)

    processed_entry = ProcessedEntry(
        image_uri=image_uri,
        plant_id=plant_id
    )
    insert_processed_entry(conn, processed_entry)


if __name__ == "__main__":

    #  Setup up variables
    config = load_database_config()
    conn = connect(config)
    days = list(range(0, 360))
    datetime_days = [datetime.today() + timedelta(days=date) for date in days]

    # Clear existing data
    clear_database = input("Would you like to clear the database (N/y)")
    if clear_database in "Yy":
        print("Clearing Database\n")
        with conn.cursor() as cursor:
            cursor.execute("""truncate image_processing.processed_entry, image_processing.raw_entry""")
            conn.commit()

    # Choose how to generate data
    print("How would you like to generate data:")
    print("1. Regular growth")
    print("2. Exponential but not destructive growth")
    print("3. Exponential destructive growth")
    valid_option = False
    while not valid_option:
        inp = input("Please input option:")

        if inp == "1":
            print("Filling database with regular growth data")
            totals_knotweed = [growth_map(x) for x in days]
            totals_rose = [growth_map(x, offset=3) for x in days]
            valid_option = True
        elif inp == "2":
            print("Filling database with exponential 'non destructive' growth data")
            totals_knotweed = [growth_map(x, growth_factor=0.1) for x in days]
            totals_rose = [growth_map(x, offset=3) for x in days]
            valid_option = True
        elif inp == "3":
            print("Filling database with exponential, destructive growth data")
            totals_knotweed = [growth_map(x, growth_factor=0.1) for x in days]
            totals_rose = [growth_map(x, growth_factor=-0.1) for x in days]
            valid_option = True
        else:
            print("Invalid input, please try again")

    # Generate Data
    for date, knotweed_count in zip(datetime_days, totals_knotweed):
        for _ in range(math.floor(knotweed_count)):
            generate_entry(conn, date, plant_id="Knotweed")

    for date, rose_count in zip(datetime_days, totals_rose):
        for _ in range(math.floor(rose_count)):
            generate_entry(conn, date, plant_id="Rose")

    print("Finished inserting data")
