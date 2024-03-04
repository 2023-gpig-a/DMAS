import numpy as np


class PlantClusterFinder:
    # Take an image of a plant
    # Detect plants in the image
    # Run each plant through a CNN feature detector -> vector 100s long
    # Perform PCA -> vector 10s long
    # Plot it in our internal graph
    # Find clusters in our graph

    def __init__(self):
        self.graph = np.ndarray(shape=())
        self.plant_detector = PlantDetector()
        self.feature_detector = FeatureDetector()
        self.principal_components = PrincipalComponents()
        self.principal_components.load("")

    def add_image(self, image: np.ndarray):
        plants = self.plant_detector.detect(image)
        for plant in plants:
            features = self.feature_detector.detect(plant)
            principal_features = self.principal_components.transform(features)
            self.graph = np.concatenate(self.graph, principal_features)

    def find_clusters(self):
        return


class PlantDetector:
    # Take an image and return a list of plants within it
    # To train this I think I will use Pl@ntNet available at: https://zenodo.org/records/4726653#.YhNbAOjMJPY
    #   Add some plants to a canvas and try to segment it
    def detect(self, image: np.ndarray):
        raise Exception("Not implemented")


class FeatureDetector:
    # Take an image and return its feature vector
    # To train this I think I will use Pl@ntNet available at: https://zenodo.org/records/4726653#.YhNbAOjMJPY
    # Take two arbitrary plants
    #   If they are in the same class, predict 1, else 0
    def detect(self, image: np.ndarray):
        raise Exception("Not implemented")


class PrincipalComponents:
    # Given the feature vectors from our dataset, calculate the principal components
    # Store these so that we can use them for future entries
    def generate_principal_components(self, features: np.ndarray):
        raise Exception("Not implemented")

    def transform(self, features: np.ndarray):
        raise Exception("Not implemented")

    def save(self):
        raise Exception("Not implemented")

    def load(self, location: str):
        raise Exception("Not implemented")
