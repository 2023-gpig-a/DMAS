import requests
import json
from pprint import pprint

from Util.Exceptions import PlantsUndetectedError


def detect(image_path: str):
    with open("plantnet_API_key", "r") as file:
        API_KEY = file.readline()

    PROJECT = "all"  # try "weurope" or "canada"
    api_endpoint = f"https://my-api.plantnet.org/v2/identify/{PROJECT}?api-key={API_KEY}"

    # TODO check leafs, fruit, and bark
    #   Check if this is able to detect different plants in an image
    data = {'organs': ['flower']}

    files = [
        ('images', (image_path,  open(image_path, 'rb'))),
    ]

    req = requests.Request('POST', url=api_endpoint, files=files, data=data)
    prepared = req.prepare()

    s = requests.Session()
    response = s.send(prepared)
    json_result = json.loads(response.text)
    if 'statusCode' in json_result.keys() and json_result['statusCode'] == 404:
        raise PlantsUndetectedError("Plant not recognised")

    return json_result["results"]


if __name__ == "__main__":
    results = detect("../Data/succulents.jpg")
    pprint(results)
