from __future__ import division
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math
from scipy.spatial.distance import pdist, squareform 
from sklearn.decomposition import PCA


from tsptw_with_ortools import Solver





# Data Generator
class DataGenerator(object):

    def __init__(self,solver):
        """Construct a DataGenerator."""
        self.solver=solver


    def solve_instance(self, sequence, tw_open, tw_width):
        start_time = time()
        demands = np.ones(TW_open[0].size)

        try: 
            # Calculate distance matrix
            dist_array = pdist(sequence)
            dist_matrix = squareform(dist_array)
            # Call OR Tools to solve instance
            tour_length, trip = self.solver.run(dist_matrix, demands, tw_open, tw_width)
        except: 
            tour_length = 'No solution found.'
            trip  = 'No solution found.'

        end_time = time()
        print 'Resolution time:', end_time - start_time
        return tour_length, trip


    def next_batch(self, batch_size, max_length, dimension, seed=0):
        """Return the next batch of the data"""
        if seed!=0:
            np.random.seed(seed)

        input_batch = [] # input sequence
        coord_batch = [] # input sequence coordinates
        TW_open = [] # input sequence t_open constraint 1
        TW_width = [] # input sequence t_width constraint 2

        for _ in range(batch_size):
            # Randomly generate city coordinates
            sequence = np.random.randint(100, size=(max_length+1, dimension))/100 # (max_length+1) cities with (dimension) random integer coordinates in [0,100[ scaled to [0,1[       Rq: +1 for depot

            # Principal Component Analysis to center & rotate coordinates
            pca = PCA(n_components=dimension) 
            sequence = pca.fit_transform(sequence)

            # TW constraint 1
            tw_open = np.random.randint(100, size=(max_length, 1))/100 # t_open random integer in [0,100[ scaled to [0,1[
            tw_open = np.concatenate(([[-1]],tw_open), axis=0) # depot opens at -1
            tw_open[::-1].sort(axis=0) # sort cities by TW Open constraint (reverse order - city contains future information)	Rq: depot = tw_open[-1], tw_width[-1] and sequence[-1]
            
            # TW constraint 2
            tw_width = (10+np.random.randint(40, size=(max_length, 1)))/100 # t_width random integer in [10,50[ scaled to [0.1,0.5[
            tw_width = np.concatenate((tw_width,[[1]]), axis=0) # depot opened for 1 (until 0)

            # Concatenate input (sorted by time)
            input_ = np.concatenate((sequence,tw_open,tw_width), axis=1)

            # Store batch
            coord_batch.append(sequence)
            TW_open.append(tw_open)
            TW_width.append(tw_width)
            input_batch.append(input_)

        return input_batch, coord_batch, TW_open, TW_width


    def single_shuffled_batch(self, batch_size, max_length, dimension, seed=0):
        if seed!=0:
            np.random.seed(seed)

        # Randomly generate single sequence of city coordinates
        sequence = np.random.randint(100, size=(max_length+1, dimension)) # (max_length+1) cities with (dimension) random integer coordinates in [0,100[ scaled to [0,1[       Rq: +1 for depot

        """
        # Principal Component Analysis
        pca = PCA(n_components=dimension) 
        sequence = pca.fit_transform(sequence)
        """

        # TW constraint 1
        tw_open = np.random.randint(100, size=(max_length, 1)) # tw_open random integer in [0,100[ scaled to [0,1[
        tw_open = np.concatenate(([[-1]],tw_open), axis=0) # depot opens at -1
        tw_open[::-1].sort(axis=0) # sort cities by TW Open constraint (reverse order - city contains future information)   Rq: depot = tw_open[-1], tw_width[-1] and sequence[-1]
        
        # TW constraint 2
        tw_width = (10+np.random.randint(40, size=(max_length, 1))) # t_width random integer in [10,50[ scaled to [0.1,0.5[
        tw_width = np.concatenate((tw_width,[[1]]), axis=0) # depot opened for 1 (until 0)

        # Concatenate input (sorted by time)
        input_ = np.concatenate((sequence,tw_open,tw_width), axis=1)

        # Store batch
        input_batch = np.tile(input_,(batch_size,1,1)) # input sequence
        coord_batch = np.tile(sequence,(batch_size,1,1)) # input sequence coordinates
        TW_open = np.tile(tw_open,(batch_size,1,1)) # input sequence t_open constraint
        TW_width = np.tile(tw_width,(batch_size,1,1)) # input sequence t_width constraint

        return input_batch, coord_batch, TW_open, TW_width



    def visualize_2D_trip(self,trip,tw_open,tw_width,time_delivery):
        #plt.figure(figsize=(30,30))
        #rcParams.update({'font.size': 22})

        # Plot cities
        colors = ['red'] # Depot is last
        for i in range(len(tw_open)-1):
            colors.append('blue')
        plt.scatter(trip[:,0], trip[:,1], color=colors)

        # Plot tour
        tour=np.array(list(range(len(trip))) + [0])
        X = trip[tour, 0]
        Y = trip[tour, 1]
        plt.plot(X, Y,"--", markersize=100)

        # Annotate cities with TW
        time_window = np.concatenate((tw_open,tw_open+tw_width),axis=1)
        for tw, (x, y) in zip(time_window,(zip(X,Y))):
            plt.annotate(tw,xy=(x, y))  

        plt.xlim(0,100)
        plt.ylim(0,100)

        plt.show()

    def indexes_to_coordinates(self,coordinates,tw_open,tw_width,trip_indexes):
        trip_coodinates = [coordinates[i] for (i,j,k) in trip_indexes]
        trip_tw_open = [tw_open[i] for (i,j,k) in trip_indexes]
        trip_tw_width = [tw_width[i] for (i,j,k) in trip_indexes]
        return np.asarray(trip_coodinates), np.asarray(trip_tw_open), np.asarray(trip_tw_width)


def example():
    locations = [[820, 760], [960, 440], [50, 50], [490, 80], [130, 70], [290, 890], [580, 300],
               [840, 390], [140, 240], [120, 390], [30, 820], [50, 100], [980, 520], [840, 250],
               [610, 590], [10, 650], [880, 510], [910, 20], [190, 320], [930, 30], [500, 930],
               [980, 140], [50, 420], [420, 90], [610, 620], [90, 970], [800, 550], [570, 690],
               [230, 150], [200, 700], [850, 600], [980, 50]]
    dist_array = pdist(locations)
    dist_matrix = squareform(dist_array)

    demands =  [0, 19, 21, 6, 19, 7, 12, 16, 6, 16, 8, 14, 21, 16, 3, 22, 18,
             19, 1, 24, 8, 12, 4, 8, 24, 24, 2, 20, 15, 2, 14, 9]

    tw_open =  [0, 508, 103, 493, 225, 531, 89,
                  565, 540, 108, 602, 466, 356, 303,
                  399, 382, 362, 521, 23, 489, 445,
                  318, 380, 55, 574, 515, 110, 310,
                  387, 491, 328, 73]

    max_length = len(demands)

    # tw_duration is the width of the time windows.
    tw_width = [2150] * max_length


    solver = Solver(max_length)
    total_distance, trip = solver.run(dist_matrix, np.asarray(demands), np.asarray(tw_open), np.asarray(tw_width))
    print "Total distance", total_distance
    print "Trip : \n", trip



if __name__ == "__main__":

    # Config
    batch_size=1
    max_length=10
    dimension=2

    # Create Solver and Data Generator
    solver = Solver(max_length)
    dataset = DataGenerator(solver)

    # Generate some data
    #input_batch, coord_batch, TW_open, TW_width = dataset.next_batch(batch_size, max_length, dimension,seed=0)
    input_batch, coord_batch, TW_open, TW_width = dataset.single_shuffled_batch(batch_size, max_length, dimension,seed=0)

    # Some nice print
    print('Input: \n',input_batch[0])

    # Solve to optimality
    tour_length, trip  = dataset.solve_instance(coord_batch[0],TW_open[0],TW_width[0])
    print('Solver tour length: \n',tour_length)
    print('Trip: \n',trip)


    # 2D plot for coord batch
    dataset.visualize_2D_trip(coord_batch[0], TW_open[0], TW_width[0], TW_open[0])

    # 2D plot for solver trip
    if type(tour_length)!=str:
        trip_coodinates, trip_tw_open, trip_tw_width = dataset.indexes_to_coordinates(coord_batch[0],TW_open[0],TW_width[0],trip)
        dataset.visualize_2D_trip(trip_coodinates, trip_tw_open, trip_tw_width, trip_tw_open)
    