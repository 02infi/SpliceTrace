import numpy as np
with open("/home/gaurav/Master_Project/Data/TPM/GSE57872_TP","rU") as filename :
	file = filename.readlines()
	header_file,lines = file[0],file[1:]
	matr_data = np.array(lines) 
	transcript_data_array = np.array([line.strip().split('\t') for line in lines])
	transcript_data_array[:,0] = np.core.defchararray.rpartition(np.array(transcript_data_array[:,0]) ,'.')[:,0]
	np.savetxt('GSE57872_TPM_M.txt',transcript_data_array,header = header_file,delimiter = '	',newline='\r',comments = '',fmt = "%s")


