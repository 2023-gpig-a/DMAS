import os
from PIL import Image


def save(image: Image, filename: str) -> str:
    # Currently in place until we have a proper s3 mock in place
    if not os.path.exists('Data/'):
        os.makedirs('Data')

    image.save(f"Data/{filename}")

    return f"Data/{filename}"