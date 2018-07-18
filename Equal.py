import numpy as np
import sys 
import pandas as pd 


def main():
	file = sys.argv
	single = pd.read_csv(file[1],sep = '\t')
	bulk = pd.read_csv(file[2],sep = '\t')
	single = np.array(single)
	bulk = np.array(bulk)
	data = np.stack((single[:,0],bulk[:,0]),axis = -1)
	boolean_value = np.isnan(data)
	new_data = data[np.logical_xor(boolean_value[:,0],boolean_value[:,1])]
	new_data[np.isnan(new_data)] = -1
	result = np.add(np.square(new_data[:,0]),new_data[:,1])
	single_over_bulk = new_data[result < 0.5]
	bulk_over_single = new_data[result > 0.5]
	print (len(single_over_bulk),len(bulk_over_single))

if __name__ == '__main__':
    main()

