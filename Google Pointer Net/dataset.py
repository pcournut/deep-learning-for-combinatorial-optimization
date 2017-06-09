import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import pdist, squareform 

#from tsp_with_ortools import Solver


# Data Generator
class DataGenerator(object):

    def __init__(self,solver):
        """Construct a DataGenerator."""
        self.solver=solver


    def solve_batch(self, coord_batch):
        opt_length_batch = []

        for sequence in coord_batch:
            # Calculate dist_matrix
            dist_array = pdist(sequence)
            dist_matrix = squareform(dist_array)
            # Calculate optimal_tour_length
            optimal_tour_length=[self.solver.run(dist_matrix)]
            opt_length_batch.append(optimal_tour_length)

        return opt_length_batch


    def next_batch(self, batch_size, max_length, dimension, seed=0):
        """Return the next batch of the data"""
        if seed!=0:
            np.random.seed(seed)

        coord_batch = []
        for _ in range(batch_size):
            # Randomly generate data
            sequence = np.random.rand(max_length, dimension).astype(np.float32) # (max_length) random numbers with (dimension) coordinates in [0,1]
            # Store batch
            coord_batch.append(sequence)

        return coord_batch


    def single_shuffled_batch(self, batch_size, max_length, dimension, seed=42):
        if seed!=0:
            np.random.seed(seed)

        # Single sequence of coordinates
        seq = np.random.rand(max_length, dimension).astype(np.float32) # (max_length) random numbers with (dimension) coordinates in [0,1]
        
        coord_batch = []
        # Randomly shuffle seq
        for _ in range(batch_size):
            # Shuffle sequence
            sequence = np.copy(seq) ##########################
            np.random.shuffle(sequence) ##########################
            # Store batch
            coord_batch.append(sequence )

        return coord_batch


    def shuffle_batch(self, coord_batch):

        coord_batch_ = []
        # Randomly shuffle seq
        for seq in coord_batch:
            # Shuffle sequence
            sequence = np.copy(seq) ##########################
            np.random.shuffle(sequence) ##########################
            # Store batch
            coord_batch_.append(sequence)

        return coord_batch_



    def visualize_2D_trip(self,trip):
        # plot 2D graph
        plt.scatter(trip[:,0], trip[:,1])
        labels=np.array(list(range(len(trip))) + [0])
        X = trip[labels, 0]
        Y = trip[labels, 1]
        plt.plot(X, Y)

        for i, (x, y) in zip(labels,(zip(X,Y))):
            plt.annotate(i,xy=(x, y))

        plt.xlim(0,1)
        plt.ylim(0,1)

        plt.show()







if __name__ == "__main__":

    # Config
    batch_size=128
    max_length=20
    dimension=2

    # Create Solver and Data Generator
    solver = Solver(max_length)
    dataset = DataGenerator(solver)

    # Next batch
    #coord_batch = dataset.next_batch(batch_size, max_length, dimension,seed=0)
    coord_batch = dataset.single_shuffled_batch(batch_size, max_length, dimension,seed=75)

    # Some nice print
    print('Coordinates: \n',coord_batch[0])

    # Solve to optimality
    opt_length_batch = dataset.solve_batch(coord_batch)
    print('Optimal tour length: \n',opt_length_batch[0])

    # 2D plot for coord batch
    dataset.visualize_2D_trip(coord_batch[0])
    dataset.visualize_2D_trip(coord_batch[1])