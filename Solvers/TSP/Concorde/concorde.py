import os 
import time
import numpy as np 

class Solver():

	def __init__(self, tsp_size):
		self.max_length = tsp_size
		self.header = """NAME : TSP instances
COMMENT : Pierre Cournut
TYPE : TSP
DIMENSION : %d
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_TYPE : TWOD_COORDS
NODE_COORD_SECTION
""" % self.max_length


	def run(self,sequence):
	
		sequence_str = ''
		for k in range(self.max_length):
			sequence_str += '%d %f %f\n' % (k, sequence[k][0], sequence[k][1])
		with open('sequence.tsp', 'w') as output_file:
		    output_file.write(self.header)
		    output_file.write(sequence_str)
		os.system("./concorde sequence.tsp")
		# os.system("./concorde sequence.tsp %s > /dev/null")
		X,Y,labels = [],[],[]
		with open('sequence.sol', 'r') as f:
			for line in f.readlines():
				tokens = line.split()
				for c in tokens:
					if int(c) != self.max_length:
						[x,y] = sequence[int(c)]
						X.append(x)
						Y.append(y)
						labels.append(int(c))
			[x,y] = sequence[0]
			X.append(x)
			Y.append(y)
		return np.concatenate((np.expand_dims(np.asarray(X),1),np.expand_dims(np.asarray(Y),1)),axis=1)







