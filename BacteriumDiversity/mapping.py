
import pandas as pd
from csv import reader
import numpy

def mapping():


    # 1. Read csv and get col names and numbers
    df = pd.read_csv('Raw_data_ordered_mapped.csv', sep=",", header=None)
#    col_names = list(df.columns)

#    df.drop(['0'], axis=1, inplace=True)
    numpy_data = df.to_numpy()

    print(numpy_data)
    for i in range(1, len(numpy_data)):
        for j in range(1, len(numpy_data[i])):
            if numpy_data[i][j] < 300:
                # print(numpy_data[i][j])
                # print("Changing")
                numpy_data[i][j] = 0


    numpy.savetxt('newData_normalized.csv', numpy_data.round(8), fmt = '%1.8f',  delimiter=',')



mapping()
