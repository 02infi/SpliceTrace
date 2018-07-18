import numpy as np
import pandas as pd 
import networkx as nx 
import seaborn as sns
import matplotlib.pyplot as plt
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
			Event_Energy = (-0.5 * Graph.nodes[each_event]['psi'] * np.sum(Imputed_list[neighbour_event,cell]) * np.float(Graph.degree(each_event)))
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


def Cell_Types_Split(data,cells):
	to_cells =[]
	Types = []
	from_cells = []
	to_cells = np.cumsum(cells)
	from_cells = 0
	from_cells = np.array(from_cells)
	from_cells = np.hstack((from_cells,to_cells[0:len(cells)-1]))
	to_cells = list(to_cells)
	from_cells = list(from_cells)
	for i in range(0,len(cells)):
		Types.append(np.array(data[from_cells[i] : to_cells[i]]))
	return Types






# export PATH=~/anaconda3/bin:$PATH "before running do export anaconda"
# minimum fraction of data should be consider for imputation 
#cells = [74,69,61] #For know cell types

data = pd.read_csv('',sep = '\t') # load the psi values from SUPPA output
alpha = 0.90 
data = np.column_stack((data.index,data.as_matrix()))
data = np.array(data)
Intermediate = Remove_NaN(data) #for single type of data
cells = [data.shape[1]-1]
IMPUTED_data = Imputation_of_missing_data(Intermediate,cells)

IMPUTED_data = np.array(Intermediate[:,1:],dtype = float) #if there is no need of imputation 

#Intermediate = Multipe_Removal_NaN(data,cells) # multiple cell types
#Impute_data = data[np.any(Intermediate > 0.80,axis = 0)] # multiple cell types
#IMPUTED_data = Imputation_of_missing_data(Impute_data,cells)
Imputed_list = np.array(IMPUTED_data) # convert the arrays into list for muliplte cell types
Imputed_list[Imputed_list > 1] = 1 # imputation can cause the value to be greater than 1
Imputed_list = np.hstack(Imputed_list)
#IMPUTED_data = np.array(IMPUTED_data)
#IMPUTED_data[IMPUTED_data > 1] = 1 # imputation can cause the value to be greater than 1


PCA_reduced_data = PCA_Dimension_reduction(Imputed_list) # reducing the dimension of the data into 2 dimension
np.savetxt("PCA_dimension_reduc_data.csv",PCA_reduced_data,delimiter = ',',fmt = '%1.4f')

TSNE_Dimension_redu_data = TSNE_Dimension_reduction(Imputed_list)
np.savetxt("TSNE_dimension_reduc_data.csv",TSNE_Dimension_redu_data,delimiter = ',',fmt = '%1.4f')

MDS_Dimension_redu_data = MDS_Dimension_reduction(Imputed_list)
np.savetxt("MDS_dimension_reduc_data.csv",MDS_Dimension_redu_data,delimiter = ',',fmt = '%1.4f')

Correlation_data = Correlation_Coeff(Imputed_list)
np.savetxt("Correlation.csv",Correlation_data,delimiter= ',',fmt = '%1.4f')


# Determining significant events from correlations value
Correlation_data = Correlation_Coeff(Imputed_list)
condlist = [Correlation_data < -0.75, Correlation_data > 0.75]
choicelist = [Correlation_data,Correlation_data]
significant_corr_data = np.select(condlist,choicelist)
events_index = np.where(np.count_nonzero(significant_corr_data,axis = 0) > 1)
#significant_events = Intermediate[events_index]
significant_psi_values = Imputed_list[events_index]


PCA_reduced_data = PCA_Dimension_reduction(significant_psi_values) # reducing the dimension of the data into 2 dimension
np.savetxt("PCA_significant_events.csv",PCA_reduced_data,delimiter = ',',fmt = '%1.4f')

TSNE_Dimension_redu_data = TSNE_Dimension_reduction(significant_psi_values)
np.savetxt("TSNE_significant_events.csv",TSNE_Dimension_redu_data,delimiter = ',',fmt = '%1.4f')

MDS_Dimension_redu_data = MDS_Dimension_reduction(significant_psi_values)
np.savetxt("MDS_significant_events.csv",MDS_Dimension_redu_data,delimiter = ',',fmt = '%1.4f')




data = PCA_reduced_data
colors = ['navy', 'turquoise', 'darkorange']
target_names = ['Mononuclear Cell','Neural Progenitor Cell','iPluripotent Cell']
from_cells = np.cumsum(cells)
to_cells = 0
to_cells = np.array(to_cells)
to_cells = np.hstack((to_cells,from_cells[0:len(cells)-1]))
from_cells = list(from_cells)
to_cells = list(to_cells)

for color,i,target in zip (colors,[0,1,2],target_names):
	plt.scatter(data[from_cells[i]:to_cells[i]],color = colors)
	print(data[from_cells[i]:to_cells[i],0])

for i in range(len()):
	plt.text(X,Y,str(i))

plt.show()




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



G  = nx.Graph() # declare the graph
G.add_weighted_edges_from(edges.T)

matrix = np.zeros((len(np.array(G.nodes())),Imputed_list.shape[1] + 1))
matrix[:,0] = np.array(G.nodes()) 
for cell in range(0,Imputed_list.shape[1]):
	matrix = query_by_value(matrix,cell+1,Graphs_edges[cell]) #list(nx.connected_component_subgraphs(Graphs[cell]))[0]

for event in range(0,matrix.shape[0]):
	plt.plot(matrix[event,1:])
plt.show()

Binary_psi = Imputed_list
Binary_psi[Binary_psi < 0.5] = 0 
Binary_psi[Binary_psi >= 0.5] = 1 


Graphs_edges = []
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
	adjacency_matrix.append(Matrix) 


#--------------------------------------------------- Degree Analysis ----------------------------------------------------------------------------------------------------------------------
G  = nx.Graph() # declare the graph
G.add_weighted_edges_from(edges.T)
G_consensus_matrix = nx.adjacency_matrix(G,weight=None).todense()
Degree_ratio_Adj_matrix = []
for cell in range(0,len(adjacency_matrix)):
	Degree_ratio_Adj_matrix.append(np.mean(np.divide(np.sum(adjacency_matrix[cell],axis = 1),np.sum(G_consensus_matrix,axis = 1))))

Degree_ratio_Adj_matrix = np.array(Degree_ratio_Adj_matrix)

for event in range(0,Degree_ratio_Adj_matrix.shape[0]):
	sns.boxplot(data = Degree_ratio_Adj_matrix[event,:])
plt.show()



#--------------------------------------------Draw Graphs ------------------------------------------------------------------------------------------------------
Data_degree_centrality = []
Mean_Degree_C = []
Mean_psi = []
Data_psi = []
Mean_Jaccard_index = []
Mean_Jaccard_index = np.array(Mean_Jaccard_index)
for cell in range(0,Imputed_list.shape[1]):
	G = Graphs_edges[cell]
	Degree_C = np.array(list(nx.degree_centrality(G).items()))[:,1]
	nodes = np.array(list(nx.get_node_attributes(G,'psi').items()))[:,1]
	Mean_Degree_C.append(np.mean(Degree_C))
	Mean_psi.append(np.mean(nodes))
	Data_psi.append(nodes)
	#Data_degree_centrality.append(Degree_C)


Data_degree_centrality = np.array(Data_degree_centrality)
Variance_events = []
for i in range(0,Data_degree_centrality.shape[1]):
	Variance_events.append(np.var())





Mean_Jaccard_index = []
Mean_Jaccard_index = np.array(list(nx.jaccard_coefficient(Graphs_edges[0])))[:,2]
for cell in range(0,Imputed_list.shape[1]):
	Mean_Jaccard_index = np.vstack((Mean_Jaccard_index,np.array(list(nx.jaccard_coefficient(Graphs_edges[cell])))[:,2]))










for cell in range(0,Imputed_list.shape[1]):
	G = Graphs_edges[cell]#list(nx.connected_component_subgraphs(Graphs[cell]))[0] 
	options = {'node_color': Data_psi[cell],'node_size': 44,'line_color': 'black','linewidths': 0,'width': 0.5}
	nx.draw(G,**options)
plt.show()



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
options = {'node_color': Imputed_list[:,1:],'node_size': 44,'line_color': 'black','linewidths': 0,'width': 0.5}




#-------------------------------------------------ISING MODEL 1D --------------------------------------------------------------------------------------------


States = Imputed_list
States = np.array(States)
a,b = States.shape
Energy_Cells = []
alpha = 0
States[States <= alpha]  = -1
States[States > alpha]  = 1
Cells_States = []

for i in range(0,b):
	Energy = Ising_1D(States[:,i],Imputed_list[:,i])
	Energy_Cells.append(Energy)
	Cells_States.append(i+1)

Energy_Cells = np.array(Energy_Cells)
colors = ['navy', 'turquoise', 'darkorange']
cells = [74,69,61]
from_cells = np.cumsum(cells)
to_cells = 0
to_cells = np.array(to_cells)
to_cells = np.hstack((to_cells,from_cells[0:len(cells)-1]))
from_cells = list(from_cells)
to_cells = list(to_cells)
target_names = ['Mononuclear Cell','Neural Progenitor Cell','iPluripotent Cell']



for color,i,target in zip (colors,[0,1,2],target_names):
	plt.scatter(np.arange(to_cells[i],from_cells[i],1),Energy_Cells[to_cells[i]:from_cells[i]],color = color,label = target)
	plt.legend(loc='upper center')
plt.title('alpha = 0.3')

plt.show()





#-------------------------------------------------ISING MODEL FOR PSEUDO 2D --------------------------------------------------------------------------------------------


def Ising_pseudo_2D(configA,configB,configC,Values):
	Energy = 0
	up = 0
	left = 0
	for i in range(0,len(configA)-1):
		S = Values[i] 
		nb = configA[i] + configB[i+1] + configC[i] + up
		Energy = Energy + (-S) * (nb)
		up = configB[i]
	i = len(configA)-1
	S = Values[i]
	nb = configA[i] + 0 + configC[i] + up
	Energy = Energy + -S * nb
	return Energy


Events,Cells = Imputed_list.shape
A = np.zeros(Events)
Energies = []

for j in range(0,Cells-2) :
	B = States[:,j]
	C = States[:,j+1]
	Energies.append(Ising_pseudo_2D(A,B,C,Imputed_list[:,j]))
	A = B
	B = C
	C = States[:,j+2]

C = np.zeros(Events)
Energies.append(Ising_pseudo_2D(A,B,C,Imputed_list[:,Cells-1]))


#--------------------------------------------------------------------------Simulation-------------------------------------------------------------

fig, ax = plt.subplots()
options = {'node_color': Data_psi[cell]*10,'node_size': 44,'line_color': 'black','linewidths': 0,'width': 0.5}


for cell in range(0,4):
	G = Graphs_edges[cell]#list(nx.connected_component_subgraphs(Graphs[cell]))[0] 
	options = {'node_color': Data_psi[cell]*10,'node_size': 44,'line_color': 'black','linewidths': 0,'width': 0.5}
	nx.draw(G,**options)
	plt.show()
	time.sleep(1)
	plt.close()  
	#ax.imshow(adjacency_matrix[i])
	#ax.set_title("frame {}".format())
	# Note that using time.sleep does *not* work here!



for i in range(0,len(Graphs_edges)):



