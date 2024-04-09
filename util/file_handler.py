import os
import uuid
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

from util.exceptions import GPSUndefinedError


def save(image: Image, filename: str) -> str:
    # TODO replace with S3
    if not os.path.exists('Data/'):
        os.makedirs('Data')

    _, ext = filename.split(".")
    filename = uuid.uuid4().hex
    image.save(f'Data/{filename}.{ext}')
    return f'Data/{filename}.{ext}'


def gps_details(image: Image) -> {}:
    exif_table = {}
    info = image.getexif()
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        exif_table[decoded] = value

    if 'GPSInfo' not in exif_table.keys():
        raise GPSUndefinedError()

    gps_info = {}
    for key in exif_table['GPSInfo'].keys():
        decode = GPSTAGS.get(key, key)
        gps_info[decode] = exif_table['GPSInfo'][key]
    return gps_info

