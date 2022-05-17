#import import_ipynb
import sys
import csv
import numpy as np
import math
#import pandas as pd

# Returns an array of the same size with values replaced by appropriate bin numbers.
# Bin ranges are found as ((SERIES_MAX - SERIES_MIN)/ NUMBER_BINS) * BIN_NUMBER + SERIES_MIN
def binner(series, num_bins, length, start):
#    series = load_data('pydump-percentage-af.data', 2)
    #print((binned))
    my_return = []
    for sect in series:
        new_series = binner_all(sect, num_bins, length, start)
        my_return.append(new_series)
    return my_return
def binner_all(series, num_bins, length, start):
    # Find the bin range.
    minimum = min(series)
    diff_range = (max(series) - minimum)/float(num_bins)

    # Create array of bin ranges.
    bins = [np.NINF]
#    bins=[minimum]
    for idx in range(1, num_bins):
        bins.append(minimum + (idx * diff_range))

    # length of 0, or any negative value, means length of the passed series.
    if length <= 0:
        length = len(series)

    # Return the newly binned series
    binned = np.digitize(series[start:start+length], bins, right=True)
#    print(binned)
    return binned



# Function to load data from vertical csv. Assumes no header.
def load_data(data_file, skip_columns):
    all_series = []

    # With open handles file opening and closing automatically. Then handle file as csv.
    with open(data_file) as z:
        f = csv.reader(open(data_file, 'rU'), delimiter=',')
        first_flag = 0

        # Go through csv line by line.
        for row in f:
            # Build allSeries according to how many columns are in the csv. 1 column = 1 series.
            if first_flag == 0:
                first_flag = 1
                for idx in range(skip_columns, len(row)):
                    all_series.append([])
                    
            # Append values to corresponding series.
            series_track = 0
            for i in range(skip_columns, len(row)):
                all_series[series_track].append(float(row[i]))
                series_track = series_track + 1
         
        return all_series
"""
# Test plot to validate code.
import matplotlib.pyplot as plt
my_list = list(range(0, 428))
series = loadData('pydump-percentage-af.data')
#print(binner(series, 10,0,0))

#print((binned))
for sect in series:
    newseries = binner(sect, 10,0,0,1)
    #plt.plot(my_list,newseries)
#plt.show()
"""
