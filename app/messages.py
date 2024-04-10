from enum import Enum
from typing import List
from pydantic import BaseModel


class StatusEnum(str, Enum):
    success = "Success"
    human_detected = "Human Detected"
    plant_not_detected = "Plant Not Detected"
    gps_undefined = "GPS Undefined"


class MessageResponse(BaseModel):
    status: StatusEnum
    message: List[str]


class PlantGrowthDataResponse(BaseModel):
    # On a given day we will see x plants, some of these will be plant_id p.
    # Each elem in plant_growth_data represents the proportion of plants identified that were class p in a given day
    # These will be sent to the LLM api to be formatted into a prompt for the LLM
    # We send one of these for each plant_id we have discovered
    plant_id: str
    plant_growth_data: List[float]
