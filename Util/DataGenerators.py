import random
from datetime import datetime
from Util.Data import RawEntry, ProcessedEntry, insert_raw_entry, insert_processed_entry, load_config, connect


def generate_rose_entry(date: datetime) -> None:
    raw_entry = RawEntry(
        latitude=[random.random(), random.random(), random.random()],
        longitude=[random.random(), random.random(), random.random()],
        image_uri="Data/rose_bush.jpg",
        date=date
    )
    insert_raw_entry(conn, raw_entry)

    processed_entry = ProcessedEntry(
        image_uri="Data/rose_bush.jpg",
        plant_id="Wichura's rose"
    )
    insert_processed_entry(conn, processed_entry)


def generate_succulent_entry(date: datetime) -> None:
    raw_entry = RawEntry(
        latitude=[random.random(), random.random(), random.random()],
        longitude=[random.random(), random.random(), random.random()],
        image_uri="Data/succulents.jpg",
        date=date
    )
    insert_raw_entry(conn, raw_entry)

    processed_entry = ProcessedEntry(
        image_uri="Data/succulents.jpg",
        plant_id="Hen and Chicks Succulent"
    )
    insert_processed_entry(conn, processed_entry)


def generate_regular_growth_data():
    # TODO
    pass


def generate_exponential_non_destructive_data():
    # TODO
    pass


def generate_destructive_data():
    # TODO
    pass


if __name__ == "__main__":

    config = load_config(filename="../database.ini")
    conn = connect(config)

    print("How would you like to generate data:")
    print("1. Regular growth")
    print("2. Exponential but not destructive growth")
    print("3. Exponential destructive growth")
    finished = False
    while not finished:
        inp = input("Please input option:")

        if inp == "1":
            print("Filling database with regular growth data")
            generate_regular_growth_data()
            finished = True
        elif inp == "2":
            print("Filling database with exponential non destructive growth data")
            generate_exponential_non_destructive_data()
            finished = True
        elif inp == "3":
            print("Filling database with exponential, destructive growth data")
            generate_destructive_data()
            finished = True
        else:
            print("Invalid input, please try again")