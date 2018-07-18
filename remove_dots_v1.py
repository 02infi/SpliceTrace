import numpy as np
import sys
path = sys.argv

with open(path[1],"rU") as filename :
	file = filename.readlines()
	header_file,lines = file[0],file[1:]
	matr_data = np.array(lines) 
	transcript_data_array = np.array([line.strip().split('\t') for line in lines])
	transcript_data_array[:,0] = np.core.defchararray.rpartition(np.array(transcript_data_array[:,0]) ,'.')[:,0]
	output = path.split('/')[-1].split('.')[0] + '_M.txt'
	np.savetxt(output,transcript_data_array,header = header_file,delimiter = '	',comments = '',fmt = "%s")
filename.closed()



