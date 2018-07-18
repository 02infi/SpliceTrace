import numpy as np
import pandas as pd 
import networkx as nx 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,MDS
from fancyimpute import KNN
from MulticoreTSNE import MulticoreTSNE as TSNE
import sys


file = sys.argv
info = pd.read_csv(file[1],sep = '\t')
TPM = pd.read_csv(file[2],sep = '\t')
file_m = '/home/gaurav' + str(file[3]) 


info_a = np.array(info)
data = []
for i in range(0,len(info_a)):
	data.append(list(info_a[i])[0].split(','))


TPM_Index = TPM.index
gene_expression = []
for i in range(0,len(data)):
	gene = np.zeros(TPM.shape[1])
	for j in range(1,len(data[i])):
		gene = gene + np.array(TPM[data[i][j] == TPM_Index])
	print (i)
	gene_expression.append([data[i][0],gene])

np.save(file_m,gene_expression)


#Gene = 0
#Gene = gene_expression[0][1]
#Index = []
#Index.append(gene_expression[0][0])
#for i in range(1,len(gene_expression)):
#	Gene = np.vstack((Gene,gene_expression[i][1]))
#	Index.append(gene_expression[i][0])







#information = []
#for i in range(0,len(info)):
##	a = list(info[i])
#	del a[1]
#	information.append(a)


# with open('/home/gaurav/information.csv', "w") as output:
#   writer = csv.writer(output, lineterminator='\n')
#    writer.writerows(information)





