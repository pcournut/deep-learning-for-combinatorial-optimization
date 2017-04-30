import numpy as np
import matplotlib.pyplot as plt

class DataGenerator(object):

    def __init__(self):
        """Construct a DataGenerator."""
        pass

    def next_batch(self, batch_size, max_length, dimension):
        """Return the next batch of the data"""
        input_batch = [] 
        for b in range(batch_size):
            sequence = np.random.rand(max_length, dimension) # N = max_length random numbers of dim = dimension
            input_batch.append(sequence)
        return input_batch


if __name__ == "__main__":
    dataset = DataGenerator()
    r = dataset.next_batch(2, 5, 1)
    print("Reader (N array of batch_size x dim): \n ", r,'\n')

    # plot 1D random numbers
    batch_0=r[1]
    points = []
    for position in batch_0:
        points.append(position[0])

    plt.plot(points)
    plt.show()

    """
    # plot 2D graph
    X,Y = [],[]
    for point in r:
        x=point[0][0]
        y=point[0][1]
        X.append(x)
        Y.append(y)

    plt.plot(r,'ro')
    plt.show()
    """
