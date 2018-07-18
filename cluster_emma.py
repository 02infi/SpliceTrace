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


# Take the transpose of the Splicedata from cells * events into events * cells 
def Transpose_array(filename):
	data = pd.read(filename,header = None).as_matrix()
	transpose = pd.DataFrame(np.asarray(data).T.tolist())
	transpose.to_csv('splice_data.csv',sep = ',')

def Remove_NaN(data,*key): # the data should not contains any label
	total_length = data.shape[1] - 1
	threshold = total_length * alpha
	if key:
		return np.divide(np.count_nonzero(~np.isnan(np.array(data[:,key[0] : key [1]],dtype = float)),axis = 1),float (key[1] + key[0] - 1)) 
	else: 
		return data[np.count_nonzero(~np.isnan(np.array(data[:,1:],dtype = float)),axis = 1) > threshold]

def Multipe_Removal_NaN(data,cells):
	type_of_cells = len(cells)
	start = 1
	end = 0
	compare_matrix = []
	for cell in cells : 
		end = end + cell
		compare_matrix.append(Remove_NaN(data,start,end))
		start = 1 + end
	return np.array(compare_matrix)


def Imputation_of_missing_data(data,cells):
	type_of_cells = len(cells)
	start = 1
	end = 0
	impute_matrix = []
	for cell in cells :
		end = end + cell + 1
		impute_matrix.append((KNN(k=int (0.5 * cell)).complete(data[:,start:end]))) 
		start = end
	return impute_matrix
	

def PCA_Dimension_reduction(data):
	n_components = 2
	pca = PCA(n_components = n_components)
	return pca.fit_transform(data.T)

def TSNE_Dimension_reduction(data):
	n_components = 2
	return TSNE(n_components = n_components).fit_transform(data.T)


def MDS_Dimension_reduction(data):
	n_components = 2
	return MDS(n_components = n_components).fit_transform(data.T)


def Correlation_Coeff(Impute_list_data):
	return np.corrcoef(Impute_list_data)


def Correlation_into_vectos(data,significant_index):
	vec = []   
	i,j = 0,0
	for i in range(0,len(data)):
		j = j + 1
		for j in range(0,len(data)):
			vec.append([i,j,data[i,j]])		
	return vec


def Psi_to_nodes(Graph,PSI):
	psi_values_nodes = np.array(Graph.nodes,dtype = int)
	for i in range(0,len(psi_values_nodes)):
		Graph.nodes[psi_values_nodes[i]]['psi'] = PSI[i]

def Ising_Network_on_networks(Graphs,Imputed_list):
	Hamiltonian = []
	Cell_Entropy = []
	clusters = []
	Centrality = [] 
	for cell in range(0,Imputed_list.shape[1]):
		Graph = Graphs[cell]
		#Graph = list(nx.connected_component_subgraphs(Graphs[cell]))[0]
		psi_values_nodes = np.array(Graph.nodes,dtype = int)
		Psi_to_nodes(Graph,Imputed_list[psi_values_nodes,cell])
		for each_event in psi_values_nodes:
			neighbour_event = np.array(list(Graph[each_event]),dtype=int)
			Event_Energy = -0.5 * Graph.nodes[each_event]['psi'] * np.sum(Imputed_list[neighbour_event,cell]) * np.float(Graph.degree(each_event))
			Hamiltonian.append(Event_Energy)
		Cell_Entropy.append(np.sum(Hamiltonian))
		Hamiltonian = []
		clusters.append(len(list(nx.connected_component_subgraphs(Graphs[cell]))))
		#clusters.append(len(Graph.nodes()))
	return np.array(Cell_Entropy),np.array(clusters)



#-------------------------------------------------ISING MODEL ON NETWORK --------------------------------------------------------------------------------------------

def Ising_Network(Graph,Imputed_list):
	Hamiltonian = []
	Cell_Entropy = []
	for cell in range(0,Imputed_list.shape[1]):
		psi_values_nodes = np.array(Graph.nodes(),dtype = int)
		Psi_to_nodes(Graph,Imputed_list[psi_values_nodes,cell])
		for each_event in psi_values_nodes:
			neighbour_event = np.array(list(Graph[each_event]),dtype=int)
			Event_Energy = -0.5 * Graph.nodes[each_event]['psi'] * np.sum(Imputed_list[neighbour_event,cell]) 
			Hamiltonian.append(Event_Energy)
		Cell_Entropy.append(np.sum(Hamiltonian))
		Hamiltonian = []
	return Cell_Entropy




def removing_nodes(G):
	H = G
	nodes_del = np.array(list(nx.get_node_attributes(H,'psi').items())) #acquring the values of nodes from the graph
	H.remove_nodes_from(nodes_del[nodes_del[:,1]<1][:,0]) # deleting the nodes which are not expressing were well. psi < 0.5
	return H 


def query_by_value(matrix,cell,G):
	values = np.array(list(nx.degree_centrality(G).items()))
	for event in range(0,len(values)):
		matrix[np.where(matrix[:,0] == values[event,0]),cell] = values[event,1]
	return matrix


def Ising_1D(configuration,data):
	L = len(configuration)
	Energy = 0
	for i in range(0,L):
		deltaE = 2*configuration[i]*configuration[(i+1)%L] + data[i] * data[(i+1)%L]
		Energy = Energy + deltaE
	return Energy


def Distribution(Graph):
	H=G
	nodes = np.array(list(nx.get_node_attributes(H,'psi').items()))
	nodes = nodes[:,1]
	return nodes 


def removing_edges(G):
	H = G
	nodes_del = np.array(list(nx.get_node_attributes(H,'psi').items()))
	removing_edges_from_nodes = nodes_del[nodes_del[:,1] < 0.5][:,0]
	H.remove_edges_from(list(H.edges(removing_edges_from_nodes)))
	G_matrix= nx.adjacency_matrix(H,weight=None).todense()
	return H,G_matrix



def UMPA_embedding(data,n,d):
	embedding = umap.UMAP(n_neighbors=n,min_dist=d,metric='correlation').fit_transform(data)
	return embedding 



def Dimension_reduction_methods(data,index,word):
	simple_PCA = PCA_Dimension_reduction(data) 
	pca = PCA(n_components = 50)
	PCA_50 = pca.fit_transform(data.T)
	mtsne = MTSNE(n_jobs=4)
	MTSNE_dim = mtsne.fit_transform(PCA_50)
	embedding_UMAP = UMPA_embedding(data,12,0.1)
	path_pca = '/users/genomics/gaurav/PSI_files/analysis/' + word + '/pca'
	path_mtsne = '/users/genomics/gaurav/PSI_files/analysis/' + word + '/mtsne'
	path_umap = '/users/genomics/gaurav/PSI_files/analysis/' + word + '/umap'
	np.save(path_pca,np.c_[index,simple_PCA])
	np.save(path_mtsne,np.c_[index,MTSNE_dim])
	np.save(path_umap,np.c_[index,embedding_UMAP])





# export PATH=~/anaconda3/bin:$PATH "before running do export anaconda"
# minimum fraction of data should be consider for imputation 
# cells = [74,69,61] #For know cell types

file = sys.argv
data = pd.read_csv(file[1],sep = '\t') # load the psi values from SUPPA output
words = file[1].split('/')[-1].split('.')[0]
alpha = 0.90
index_name = np.array(data.columns)
data = np.column_stack((data.index,data.as_matrix()))
data = np.array(data)
Intermediate = Remove_NaN(data) #for single type of data
cells = [data.shape[1]-1]
IMPUTED_data = Imputation_of_missing_data(Intermediate,cells)
Imputed_list = np.array(IMPUTED_data) # convert the arrays into list for muliplte cell types
Imputed_list[Imputed_list > 1] = 1 # imputation can cause the value to be greater than 1
Imputed_list = np.hstack(Imputed_list)

# Determining significant events from correlations value
Correlation_data = Correlation_Coeff(Imputed_list)
condlist = [Correlation_data < -0.5, Correlation_data > 0.5]
choicelist = [Correlation_data,Correlation_data]
significant_corr_data = np.select(condlist,choicelist)
events_index = np.where(np.count_nonzero(significant_corr_data,axis = 0) > 1)
#significant_events = Intermediate[events_index]
significant_psi_values = Imputed_list[events_index]



# Extraction of the significant correlation values from correlation matrix

lower_matrix = np.tril(significant_corr_data,-1) # making elements of non lower matrix to zero
y_matrix,x_matrix = np.nonzero(lower_matrix) # extracting the indices of lower matrix by checking elements with zero
correlation_for_edges = lower_matrix[y_matrix,x_matrix]
edges = np.vstack((y_matrix,x_matrix,lower_matrix[y_matrix,x_matrix]))

# Creating network of the significant splicing events using NetworkX graph library 


G  = nx.Graph()                    # declare the graph
G.add_weighted_edges_from(edges.T) # Add the nodes and edges with respective weights
significant_psi_values = Imputed_list[events_index] # psi value for each nodes 
psi_values_nodes = np.array(G.nodes,dtype = int) # Imputed_list[psi_values_nodes,0] implies the significant psi values for cell 0 
#Psi_to_nodes(G,Imputed_list[psi_values_nodes,1]) # 

#-----------------------------------------------------GRAPH ANALYSIS----------------------------------------------------------------------------------------------------------
Graphs_edges = []
Graphs_edges.append(G)
Graphs_nodes = []
adjacency_matrix = []
degree_ratio = []
for cell in range(0,Imputed_list.shape[1]):
	G  = nx.Graph()# declare the graph
	Matrix = []
	G.add_weighted_edges_from(edges.T)
	psi_values_nodes = np.array(G.nodes,dtype = int) 
	Psi_to_nodes(G,Imputed_list[psi_values_nodes,cell])
	G,Matrix = removing_edges(G)
	Graphs_edges.append(G)
	graph_file = 'Graph_'+str(cell)+'.gexf'
	nx.write_gexf(G,graph_file)
	#filename = 'Cell' + str(cell) + '.pickle.gz'
	#nx.write_gpickle(G, filename)

filename = '/users/genomics/gaurav/PSI_files/analysis/' + words +'/Graph_'+words+'.gpickle'
nx.write_gpickle(Graphs_edges,filename)
	#adjacency_matrix.append(Matrix) 




matrix = np.zeros((len(np.array(G.nodes())),Imputed_list.shape[1] + 1))
matrix[:,0] = np.array(G.nodes()) 
for cell in range(0,Imputed_list.shape[1]):
	matrix = query_by_value(matrix,cell+1,Graphs_edges[cell+1]) #list(nx.connected_component_subgraphs(Graphs[cell]))[0]

#--------------------------------------------Draw Graphs ------------------------------------------------------------------------------------------------------
Data_degree_centrality = []
Mean_Degree_C = []
Mean_psi = []
Data_psi = []
Mean_Jaccard_index = []
Mean_Jaccard_index = np.array(Mean_Jaccard_index)
for cell in range(0,Imputed_list.shape[1]):
	G = Graphs_edges[cell+1]
	Degree_C = np.array(list(nx.degree_centrality(G).items()))[:,1]
	nodes = np.array(list(nx.get_node_attributes(G,'psi').items()))[:,1]
	Mean_Degree_C.append(np.mean(Degree_C))
	Mean_psi.append(np.mean(nodes))
	#Data_psi.append(nodes)
	#Data_degree_centrality.append(Degree_C)


#--------------------------------------------------- Degree Analysis ----------------------------------------------------------------------------------------------------------------------
G  = nx.Graph() # declare the graph
G.add_weighted_edges_from(edges.T)
G_consensus_matrix = nx.adjacency_matrix(G,weight=None).todense()
Degree_ratio_Adj_matrix = []
for cell in range(0,len(adjacency_matrix)):
	Degree_ratio_Adj_matrix.append(np.mean(np.divide(np.sum(adjacency_matrix[cell],axis = 1),np.sum(G_consensus_matrix,axis = 1))))


Dimension_reduction_methods(Imputed_list,index,words)










