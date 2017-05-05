import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform 

import networkx as nx


class DataGenerator(object):

    def __init__(self):
        """Construct a DataGenerator."""
        pass

    def next_batch(self, batch_size, max_length, dimension):
        """Return the next batch of the data"""
        coord_batch = []
        dist_batch = []
        input_batch = []
        K=dimension+1 # for Nearest Neighbours (requires max_length-1 >= dimension +1 ie. max_length > dimension +1)

        for b in range(batch_size):
            # Random sequence of coordinates
            sequence = np.random.rand(max_length, dimension) # N = max_length random numbers of dim = dimension

            # Calculate dist_matrix
            dist_array = pdist(sequence)
            dist_matrix = squareform(dist_array)

            # New input representation (coord, KNN_index, KNN_dist)
            new_input=[]
            for i,line in enumerate(dist_matrix):
                i_KNN=np.argsort(line)[1:][:K]
                new_input.append(np.concatenate((sequence[i], i_KNN, line[i_KNN]),axis=0))
            new_input = np.stack(new_input,axis=0)

            # Define graph
            g=nx.Graph()
            g.add_nodes_from(range(max_length))
            for i in range(max_length):
                for j in range(dimension,K+dimension):
                    g.add_edge(i,new_input[i][j],w='%.2f'%(new_input[i][j+K]))

            # Add excess degree feature (hub vs. isolated city)
            excess_degree = [[g.degree(i) - K] for i in range(max_length)]
            temp=[]
            for i in range(max_length):
                temp.append(np.concatenate((new_input[i],excess_degree[i])))
            new_input = np.stack(temp,axis=0)

            # Store batch
            coord_batch.append(sequence)
            dist_batch.append(dist_matrix)
            input_batch.append(new_input)

        return coord_batch, dist_batch, input_batch


    def visualize_KNN_graph(self,input_sequence,dimension,tour={0: False, 1: [0,4,2,3,1]}):
        seq_length=input_sequence.shape[0]
        K=dimension+1 # Minimal description in d space

        # Define graph
        g=nx.Graph()
        g.add_nodes_from(range(seq_length))
        for i in range(seq_length):
            for j in range(dimension, K+dimension):
                g.add_edge(i,input_sequence[i][j],w='%.2f'%(input_sequence[i][j+K]))

        # Find central nodes (high degree)
        hubs = [i for i in range(seq_length) if g.degree(i) > K]

        # Plot graph
        pos=nx.spectral_layout(g) # positions for all nodes
        nx.draw_networkx_nodes(g,pos=pos)
        nx.draw_networkx_nodes(g,pos=pos, nodelist=hubs, node_color='g')
        nx.draw_networkx_labels(g,pos=pos)
        nx.draw_networkx_edges(g,pos=pos)
        nx.draw_networkx_edge_labels(g,pos=pos, rotate=False)
        
        # (Optional) Add tour
        if tour[0]==True: 
            # Get sequence of edges visited
            path=[]
            for i in range(len(tour[1])-1):
                path.append((tour[1][i],tour[1][i+1]))
            path.append((tour[1][len(tour[1])-1],tour[1][0]))

            # Extend tour to non visited cities
            non_visited_cities=[city for city in range(seq_length) if city not in tour[1]]
            for city in non_visited_cities:
                path.append((tour[1][0],city))
                path.append((city,tour[1][0]))

            nx.draw_networkx_edges(g,pos=pos,edgelist=path, width=2, edge_color='b')

        plt.show()


if __name__ == "__main__":
    batch_size=3
    max_length=8
    dimension=2

    dataset = DataGenerator()
    coord_batch, dist_batch, input_batch = dataset.next_batch(batch_size, max_length, dimension)

    for b in range(batch_size):
        print('Sequence of coordinates: \n',coord_batch[b])
        print('Distance matrix: \n',dist_batch[b])
        print('Sequence input: \n',input_batch[b])
        print('\n')
        dataset.visualize_KNN_graph(input_batch[b],dimension)

        # plot 2D graph
        X,Y,labels = [],[],[]
        for i,point in enumerate(coord_batch[b]):
            x=point[0]
            y=point[1]
            labels.append(i)
            X.append(x)
            Y.append(y)
        plt.plot(X,Y,'ro')
        for i, (x, y) in enumerate(zip(X,Y)):
            plt.annotate(i,xy=(x, y))
        plt.show()