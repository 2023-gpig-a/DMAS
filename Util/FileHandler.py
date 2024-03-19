import os
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS


def save(image: Image, filename: str) -> str:
    # Currently in place until we have a proper s3 mock in place
    if not os.path.exists('Data/'):
        os.makedirs('Data')

    image.save(f"Data/{filename}")

    return f"Data/{filename}"


def gps_details(image_file_path) -> {}:
    exif_table = {}
    image = Image.open(image_file_path)
    info = image.getexif()
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        exif_table[decoded] = value

    if len(exif_table['GPSInfo'].keys()) == 0:
        return {
            'GPSLatitudeRef': 'N',
            'GPSLatitude': (0, 0, 0),
            'GPSLongitudeRef': 'W',
            'GPSLongitude': (0, 0, 0)
        }

    gps_info = {}
    for key in exif_table['GPSInfo'].keys():
        decode = GPSTAGS.get(key, key)
        gps_info[decode] = exif_table['GPSInfo'][key]
    return gps_info

