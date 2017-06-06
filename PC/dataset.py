import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx

from scipy.spatial.distance import pdist, squareform 
from sklearn import manifold
from sklearn.decomposition import PCA
from solver import Solver


# Graph Transformation
def center_scale_sequence(seq):
    mean = seq-seq.mean(axis=0)
    deviation = np.sum(mean.std(axis=0)**2)**0.5
    return (1/deviation)*mean

def furthest_point_first(seq): # and reindex
    furthest_index=np.argmax(np.sum(seq**2,axis=1))
    seq_order=[ (furthest_index+i)% len(seq) for i in range(len(seq))]
    return seq[seq_order]

def rotate_sequence(seq):
    x_0,y_0 = seq[0]
    d = (x_0**2+y_0**2)**0.5
    cos_ = x_0/d
    sin_ = -y_0/d
    rotated_seq = []
    for point in seq :
        rotated_seq.append([point[0]*cos_-point[1]*sin_ , point[0]*sin_+point[1]*cos_])
    return np.stack(rotated_seq,axis=0)

def nearest_neighbor_policy(seq):
    dist_array = pdist(seq)
    dist_matrix = squareform(dist_array)

    ii=0
    seq_order=[0]
    sup = np.max(dist_matrix)
    while len(seq_order)!=len(seq):
        c_copy = np.copy(dist_matrix[ii])
        for j in seq_order:
            c_copy[j]=sup
        nearest_neighbor_i = np.argmin(c_copy,axis=0)
        seq_order.append(nearest_neighbor_i)
        ii=nearest_neighbor_i

    return seq[seq_order]






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


    def next_batch(self, batch_size, max_length, dimension, scale, n_components=5, seed=0, knn_policy=False):
        if seed!=0:
            np.random.seed(seed)

        """Return the next batch of the data"""
        coord_batch = []
        dist_batch = []
        init_length_batch = []
        input_batch = []

        # Randomly access data
        for _ in range(batch_size):

            # Preprocess sequence of coordinates
            sequence = np.random.rand(max_length, dimension) # (max_length) random numbers with (dimension) coordinates in [0,1]
            sequence = center_scale_sequence(sequence) # Center scale data
            sequence = furthest_point_first(sequence) # Set city_0 to the furthest point
            sequence = scale*sequence
            #sequence = rotate_sequence(sequence) # Rotate sequence to [0,x)=[0,city_0)
            if knn_policy==True:
                sequence = nearest_neighbor_policy(sequence) # Permute sequence // nearest neighbor policy

            # Calculate dist_matrix
            dist_array = pdist(sequence)
            dist_matrix = squareform(dist_array)

            # Calculate init_tour_length
            init_tour_length=0
            for i,city in enumerate(dist_matrix):
                if i!=len(dist_matrix)-1:
                    init_tour_length+=city[i+1]
                else:
                    init_tour_length+=city[0]
            init_tour_length=[init_tour_length]

            # Preprocess dist_matrix (center + kPCA + scale) ## CENTER ./ WHICH AXIS ?? SCALE AFTER PCA ??
            new_input=dist_matrix-dist_matrix.mean(axis=1)
            pca = PCA(n_components=n_components)
            new_input = pca.fit_transform(new_input)
            #new_input = new_input / new_input.std(axis=0)
            #print('\n Explained variance:',np.cumsum(pca.explained_variance_ratio_)) # Check here for value of k

            # Store batch
            coord_batch.append(sequence)
            dist_batch.append(dist_matrix)
            init_length_batch.append(init_tour_length)
            input_batch.append(new_input)

        return coord_batch, dist_batch, input_batch, init_length_batch

        
    def batch_from_seq(self, trips, n_components=4):

        """Return the next batch of the data from previous trips"""
        input_batch = []
        init_length_batch = []

        for sequence in trips:

            # Calculate dist_matrix
            dist_array = pdist(sequence)
            dist_matrix = squareform(dist_array)

            # Calculate init_tour_length
            init_tour_length=0
            for i,city in enumerate(dist_matrix):
                if i!=len(dist_matrix)-1:
                    init_tour_length+=city[i+1]
                else:
                    init_tour_length+=city[0]
            init_tour_length=[init_tour_length]

            # Preprocess dist_matrix (center + kPCA + scale)
            new_input=dist_matrix-dist_matrix.mean(axis=1)
            pca = PCA(n_components=n_components)
            new_input = pca.fit_transform(new_input)
            #new_input = new_input / new_input.std(axis=0)

            # Store batch
            init_length_batch.append(init_tour_length)
            input_batch.append(new_input)

        return input_batch, init_length_batch


    def single_shuffled_batch(self, batch_size, max_length, dimension, n_components=4,seed=0):
        if seed!=0:
            np.random.seed(seed)

        # Single sequence of coordinates
        seq = np.random.rand(max_length, dimension) # (max_length) random numbers with (dimension) coordinates in [0,1]
        seq = center_scale_sequence(seq) # Center scale data

        coord_batch = []
        dist_batch = []
        init_length_batch = []
        input_batch = []

        # Randomly access data
        for _ in range(batch_size):

            # Shuffle sequence
            sequence = np.copy(seq)
            np.random.shuffle(sequence)
            sequence = furthest_point_first(sequence) # Set city_0 to the furthest point
            #sequence = rotate_sequence(sequence) # Rotate sequence to [0,x)=[0,city_0)

            # Calculate dist_matrix
            dist_array = pdist(sequence)
            dist_matrix = squareform(dist_array)

            # Calculate init_tour_length
            init_tour_length=0
            for i,city in enumerate(dist_matrix):
                if i!=len(dist_matrix)-1:
                    init_tour_length+=city[i+1]
                else:
                    init_tour_length+=city[0]
            init_tour_length=[init_tour_length]

            # Preprocess dist_matrix (center + kPCA + scale) ## CENTER ./ WHICH AXIS ?? SCALE AFTER PCA ??
            new_input=dist_matrix-dist_matrix.mean(axis=1)
            pca = PCA(n_components=n_components) 
            new_input = pca.fit_transform(new_input)
            #new_input = new_input / new_input.std(axis=0)
            #print('\n Explained variance:',np.cumsum(pca.explained_variance_ratio_)) # Check here for value of k

            # Store batch
            coord_batch.append(sequence)
            dist_batch.append(dist_matrix)
            init_length_batch.append(init_tour_length)
            input_batch.append(new_input)

        return coord_batch, dist_batch, input_batch, init_length_batch


    def visualize_2D_trip(self,trip):
        # plot 2D graph
        X,Y,labels = [],[],[]
        for i,point in enumerate(trip):
            x=point[0]
            y=point[1]
            labels.append(i)
            X.append(x)
            Y.append(y)
        X.append(trip[0][0]) # come back to start
        Y.append(trip[0][1])
        labels.append(0)

        plt.plot(X,Y)
        for i, (x, y) in zip(labels,(zip(X,Y))):
            plt.annotate(i,xy=(x, y))

        plt.xlim(-2,2)
        plt.ylim(-2,2)

        plt.show()







if __name__ == "__main__":

    # Config
    batch_size=32
    max_length=10
    dimension=2
    n_components=5 # accounts for more than 98% of dist matrix energy

    # Create Solver and Data Generator
    solver = Solver(max_length)
    dataset = DataGenerator(solver)

    # Next batch
    # coord_batch, dist_batch, input_batch, init_length_batch = dataset.next_batch(batch_size, max_length, dimension, n_components=n_components,seed=0)
    coord_batch, dist_batch, input_batch, init_length_batch = dataset.single_shuffled_batch(batch_size, max_length, dimension, n_components=4,seed=0)

    # Some nice print
    #print('Coordinates: \n',coord_batch[0])
    #print('Distance matrix: \n',dist_batch[0])
    #print('NNet input: \n',input_batch[0])
    print 'Maximum initial tour length: \n', np.max(init_length_batch)
    print 'Minimum initial tour length: \n', np.min(init_length_batch)

    # Solve to optimality
    opt_length_batch = dataset.solve_batch(coord_batch)
    print 'Optimal tour length: \n', opt_length_batch[0][0]

    # 2D plot for coord batch
    j = np.argmin(init_length_batch)
    dataset.visualize_2D_trip(coord_batch[j])

    # 3D plot for input batch (after PCA)
    """
    X = [input_batch[0][i][0] for i in range(max_length)]
    Y = [input_batch[0][i][1] for i in range(max_length)]
    Z = [input_batch[0][i][2] for i in range(max_length)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=X, ys=Y,zs=Z)

    for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        label = '%d' % i
        ax.text(x, y, z, label)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()
    """