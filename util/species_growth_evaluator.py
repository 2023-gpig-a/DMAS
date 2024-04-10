import numpy as np
import matplotlib.pyplot as plt


class SpeciesGrowthEvaluator:
    """
    Our raw video feed should be converted to a 2D array of strings, these are maps
    Each cell in the array should identify a plant

    e.g: [[p, r, p],[p, r, r],[r, r, r]]

    Key: p = peonies, r = roses


    We can then use the SpeciesGrowthEvaluator to track if plants are growing in quantity
    The maps need to be fed to the growth evaluator in chronological order and need to be the same size


    Possible Improvements:
        1. It would be nicer if we could check for clusters of plants. e.g. centered on cell x,y there is a large cluster of 'p'.
        2. If we plan on displaying these graphs to users they should be prettier.
    """
    def __init__(self):
        self.maps = []
        self.species_data = {}

    def add(self, new_map: np.ndarray) -> None:
        """
        Add a new map to the SpeciesGrowthEvaluator
            Parameters:
                new_map (np.ndarray): 2D string array, each cell represents a region of the forest
        """
        if len(self.maps) != 0 and new_map.size != self.maps[0].size:
            raise ValueError(f"Invalid size of map added, other:{new_map.size} != {self.maps[0].size}")

        self.maps.append(new_map)

    def __str__(self) -> str:
        return self.maps.__str__()

    def display_maps(self) -> None:
        """
        Create and show Pyplot display of the maps stored in the SpeciesGrowthEvaluator
        """
        figure = plt.figure(figsize=(5, 2))
        figure.suptitle("Graph Showing Species Presence Variation Over Time")
        for i in range(len(self.maps)):
            ax = figure.add_subplot(1, len(self.maps), i + 1)
            ax.set_title(f"t={i}")
            plt.axis("off")
            plt.imshow(self.maps[i])
        plt.tight_layout()
        plt.show()

    def evaluate(self, print_stats: bool = False):
        """
        Generate statistics about the maps in the Evaluator,

        These are stored in SpeciesGrowthEvaluator.species_data


            species_data[s][Count]: An int list where index i represents the count of string s in map[i]
        """
        unique_species = np.unique(self.maps).tolist()
        for species in unique_species:

            self.species_data[species] = {
                "count": []
            }

            # Count
            for i in range(len(self.maps)):
                count_of_species = np.count_nonzero(self.maps[i] == species)
                self.species_data[species]["count"].append(count_of_species)
                if print_stats:
                    print(f"at time t={i}, count of {species} = {count_of_species}")


if __name__ == "__main__":
    import random

    # Function to simulate the spread of growth of a plant
    def spread_value(arr: np.ndarray, target: int, grow_chance: float) -> np.ndarray:
        out = arr.copy()
        for i, row in enumerate(arr):
            for j, val in enumerate(row):
                if val == target:
                    if i > 0 and random.random() < grow_chance:
                        out[i - 1][j] = val
                    if i < len(arr) - 1 and random.random() < grow_chance:
                        out[i + 1][j] = val
                    if j > 0 and random.random() < grow_chance:
                        out[i][j - 1] = val
                    if j < len(arr[0]) - 1 and random.random() < grow_chance:
                        out[i][j + 1] = val
        return out

    # Create an evaluator and use our test data on it
    evaluator = SpeciesGrowthEvaluator()
    test_map = np.random.randint(low=0, high=3, size=(50, 50))
    for _ in range(4):
        evaluator.add(test_map)
        test_map = spread_value(test_map, 2, 0.2)
    evaluator.display_maps()
    evaluator.evaluate()

    # Show the growth of each species over time
    figure = plt.figure(figsize=(5, 2 * len(evaluator.species_data.keys())))
    figure.suptitle("Graph Showing Species Growth With Respect To Time")
    for i, species in enumerate(evaluator.species_data.keys()):
        ax = figure.add_subplot(len(evaluator.species_data.keys()), 1, i + 1)
        ax.set_title(f"Species: {species}")
        plt.ylabel("Count")
        plt.xlabel("Time")
        plt.plot(range(len(evaluator.maps)), evaluator.species_data[species]["count"])
    plt.tight_layout()
    plt.show()
