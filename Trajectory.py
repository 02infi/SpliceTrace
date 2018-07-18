import numpy as np
import pickle
import sys 
import pandas as pd 
import networkx as nx 
import seaborn as sns
import matplotlib.pyplot as plt
import hdbscan 
import umap
from MulticoreTSNE import MulticoreTSNE as MTSNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,MDS
from fancyimpute import KNN



def Network_transition(GraphA,GraphB):
	Ajacent_matrix_A = nx.adjacency_matrix(GraphA,weight = None).todense()
	Ajacent_matrix_B = nx.adjacency_matrix(GraphB,weight = None).todense()
	return (np.linalg.matrix_rank(np.logical_xor(Ajacent_matrix_A,Ajacent_matrix_B)*1))



def Distance_metric(Graphs):
	length = len(Graphs)
	Distance = np.zeros((length,length))
	for i in range(0,length):
		for j in range(0,length):
			Distance[i,j] = Network_transition(Graphs[i],Graphs[j])
	return Distance


def Draw_Spanning_Tree(coordinates,Distance):
	G = nx.Graph()
	lower_matrix = np.tril(Distance,-1) # making elements of non lower matrix to zero
	y_matrix,x_matrix = np.nonzero(lower_matrix) # extracting the indices of lower matrix by checking elements with zero
	edges = np.vstack((y_matrix,x_matrix,lower_matrix[y_matrix,x_matrix]))
	G.add_weighted_edges_from(edges.T)
	pos = dict(zip(range(len(coordinates)),coordinates))
	for n, p in pos.items():
		G.node[n]['pos'] = p
	min_span_tree = nx.minimum_spanning_tree(G,weight = 'weight',algorithm = 'prim')
	return G,pos 




file = sys.argv
Graphs  = nx.read_gpickle(file[1])
Distance = Distance_metric(Graphs[1:])
coordinates = np.load(file[2])[:,1:]
trajectory = Draw_Spanning_Tree(coordinates,Distance)
nx.write_gpickle(trajectory[0], "test.gpickle")




