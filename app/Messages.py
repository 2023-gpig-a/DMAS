from enum import Enum
from typing import List

from pydantic import BaseModel


class StatusEnum(str, Enum):
    success = "Success"
    human_detected = "Human Detected"


class MessageResponse(BaseModel):
    status: StatusEnum
    message: str


class PlantGrowthDataResponse(BaseModel):
    # On a given day we will see x plants, some of these will be plant_id p.
    # We return a list where each element i, represents the proportion of the plants identified that were plant_id p
    plant_id: str
    plant_growth_data: List[float]
