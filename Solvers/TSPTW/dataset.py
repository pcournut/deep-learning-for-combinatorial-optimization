import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import pdist, squareform 

from tsptw_with_ortools import Solver


# Data Generator
class DataGenerator(object):

    def __init__(self,solver):
        """Construct a DataGenerator."""
        self.solver=solver


    def solve_batch(self, dist_matrix_batch, start_times_batch, end_times_batch):
        tour_length_batch = []
        trip_batch = []

        demands = np.ones(start_times_batch[0].size)

        for dist_matrix, start_times, end_times in zip(dist_matrix_batch, start_times_batch, end_times_batch):
            # Calculate optimal_tour_length
            
            """
            tour_length, trip = self.solver.run(dist_matrix, demands, start_times, end_times)
            tour_length_batch.append(tour_length)
            trip_batch.append(trip)            
            """
            try:
                tour_length, trip = self.solver.run(dist_matrix, demands, start_times, end_times)
                tour_length_batch.append(tour_length)
                trip_batch.append(trip)
            except:
                tour_length_batch.append('No solution found.')
                trip_batch.append('No solution found.')
            

        return tour_length_batch, trip_batch                

    def next_batch(self, batch_size, max_length, dimension, tw_duration, seed=0):
        """Return the next batch of the data"""
        if seed!=0:
            np.random.seed(seed)

        coord_batch = []
        dist_matrix_batch = []
        start_times_batch = []
        end_times_batch = []
        for _ in range(batch_size):
            # Randomly generate data
            sequence = np.random.rand(max_length, dimension).astype(np.float32) # (max_length) random numbers with (dimension) coordinates in [0,1]
            dist_array = pdist(sequence)
            dist_matrix = squareform(dist_array)
            start_times = np.random.randint(0,5,max_length).astype(np.int64)
            start_times[0] = 0
            #start_times 
            end_times = [t+tw_duration for t in start_times]
            # Store batch
            coord_batch.append(sequence)
            dist_matrix_batch.append(dist_matrix)
            start_times_batch.append(start_times)
            end_times_batch.append(end_times)

        return coord_batch, dist_matrix_batch, start_times_batch, end_times_batch


    def single_shuffled_batch(self, batch_size, max_length, dimension, tw_duration, seed=42):
        if seed!=0:
            np.random.seed(seed)

        # Single sequence of coordinates
        seq = np.random.rand(max_length, dimension).astype(np.float32) # (max_length) random numbers with (dimension) coordinates in [0,1]
        start_t = np.random.randint(0,5,max_length).astype(np.float32)
        end_t = [t+tw_duration for t in start_t]
        
        coord_batch = []
        dist_matrix_batch
        start_times_batch = []
        end_times_batch = []
        # Randomly shuffle seq
        for _ in range(batch_size):
            # Shuffle list indexes
            index = range(max_length)
            index_shuf = np.random.shuffle(index)
            # Shuffle sequence
            sequence = [seq(i) for i in index_shuf]
            dist_array = pdist(sequence)
            dist_matrix = squareform(dist_array)
            # Shuffle time windows
            start_times = [start_t(i) for i in index_shuf]
            end_times = [end_t(i) for i in index_shuf]
            # Store batch
            coord_batch.append(sequence)
            dist_matrix_batch.append(dist_matrix)
            start_times_batch.append(start_times)
            end_times_batch.append(end_times)

        return coord_batch, dist_matrix_batch, start_times_batch, end_times_batch


    def shuffle_batch(self, coord_batch, start_times_batch, end_times_batch):

        coord_batch_ = []
        dist_matrix_batch = []
        start_times_batch_ = []
        end_times_batch_ = []
        # Randomly shuffle seq
        for seq, (st, et) in zip(coord_batch,zip(start_times_batch,end_times_batch)):
            # Shuffle list indexes
            index = range(len(coord_batch))
            index_shuf = np.random.shuffle(index)
            # Shuffle sequence
            sequence = [seq(i) for i in index_shuf]
            dist_array = pdist(sequence)
            dist_matrix = squareform(dist_array)
            # Shuffle time windows
            start_times = [st(i) for i in index_shuf]
            end_times = [et(i) for i in index_shuf]
            # Store batch
            coord_batch_.append(sequence)
            dist_matrix_batch.append(dist_matrix)
            start_times_batch_.append(start_times)
            end_times_batch_.append(end_times)

        return coord_batch_, dist_matrix_batch, start_times_batch_, end_times_batch_


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

    def indexes_to_coordinates(self,coord_batch,trip_indexes):
        trip = [coord_batch[i] for (i,j,k) in trip_indexes]
        return np.asarray(trip)




def example():
    locations = [[820, 760], [960, 440], [500, 50], [490, 80], [130, 70], [290, 890], [580, 300],
               [840, 390], [140, 240], [120, 390], [30, 820], [50, 100], [980, 520], [840, 250],
               [610, 590], [10, 650], [880, 510], [910, 20], [190, 320], [930, 30], [500, 930],
               [980, 140], [50, 420], [420, 90], [610, 620], [90, 970], [800, 550], [570, 690],
               [230, 150], [200, 700], [850, 600], [980, 50]]
    dist_array = pdist(locations)
    dist_matrix = squareform(dist_array)

    demands =  [0, 19, 21, 6, 19, 7, 12, 16, 6, 16, 8, 14, 21, 16, 3, 22, 18,
             19, 1, 24, 8, 12, 4, 8, 24, 24, 2, 20, 15, 2, 14, 9]

    start_times =  [0, 508, 103, 493, 225, 531, 89,
                  565, 540, 108, 602, 466, 356, 303,
                  399, 382, 362, 521, 23, 489, 445,
                  318, 380, 55, 574, 515, 110, 310,
                  387, 491, 328, 73]

    # tw_duration is the width of the time windows.
    tw_duration = 2150

    # In this example, the width is the same at each location, so we define the end times to be
    # start times + tw_duration. For problems in which the time window widths vary by location,
    # you can explicitly define the list of end_times, as we have done for start_times.
    end_times = [0] * len(start_times)

    for i in range(len(start_times)):
        end_times[i] = start_times[i] + tw_duration

    solver = Solver()
    total_distance, trip = solver.run(dist_matrix, np.asarray(demands), np.asarray(start_times), np.asarray(end_times))
    print "Total distance", total_distance
    print "Trip : \n", trip


if __name__ == "__main__":

    # Config
    batch_size=32
    max_length=10
    dimension=2
    tw_duration = 1

    # Create Solver and Data Generator
    solver = Solver()
    dataset = DataGenerator(solver)

    # Next batch
    coord_batch, dist_matrix_batch, start_times_batch, end_times_batch = dataset.next_batch(batch_size, max_length, dimension, tw_duration, seed=0)
    #coord_batch = dataset.single_shuffled_batch(batch_size, max_length, dimension,seed=75)

    # Some nice print
    print 'Coordinates: \n',coord_batch[0]
    print 'Time windows: \n', zip(start_times_batch[0], end_times_batch[0])

    # Solve to optimality
    tour_length_batch, trip_batch = dataset.solve_batch(dist_matrix_batch, start_times_batch, end_times_batch)
    trip_coord = dataset.indexes_to_coordinates(coord_batch[0],trip_batch[0])
    print 'Tour length:',tour_length_batch[0]
    print 'Trip: \n',trip_batch[0]
    print 'Trip coordinates \n',trip_coord


    # 2D plot for coord batch
    dataset.visualize_2D_trip(coord_batch[0])
    dataset.visualize_2D_trip(trip_coord)
    #dataset.visualize_2D_trip(trip_batch[0])
    #dataset.visualize_2D_trip(coord_batch[1])