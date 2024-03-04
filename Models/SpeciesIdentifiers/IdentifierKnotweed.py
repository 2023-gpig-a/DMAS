import random

from Util.Data import ProcessedEntry


class IdentifierKnotweed:
    def __init__(self):
        self.model = [0, 1]
        
    def evaluate(self, images: [ProcessedEntry]):
        """
        :param: list of processed entries belonging with the same plant_id
        :return: whether this series of processed entries is a subset of all images containing japanese knotweed
        """
        return random.choice(self.model)
