from enum import Enum


class IdentifiedSpecies(Enum):
    # Each species here requires a model that can detect them from a series of images
    Unidentified = 0
    JapaneseKnotweed = 1
