import os
import time
import shutil
import urllib.request
import argparse
from binner import *
from build_hash import *
import csv
#likeABoss
from cal_p_current import *
#from get_all_graphs import *
from tuple_time_series import *
import phi_params as conf
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import numpy as np
from ica import compute_ica
from numpy.linalg import LinAlgError


# Function that takes two lists as inputs and returns a list of the set union of the two lists
def Union(lst1, lst2):
    if (type(lst1) == list and type(lst2) == list):
        final_lst = list(set(lst1)|set(lst2))
    elif type(lst1) == list:
        final_lst = list(set(lst1)|set([lst2]))
    elif type(lst2) == list:
        final_lst = list(set([lst1])|set(lst2))
    else:
        final_lst = list(set([lst1])|set([lst2]))
    return final_lst 

# Function that takes two lists as inputs and returns a list of the set difference A-B of the two lists
def indexdiff(A, B):
    if len(A) == 0:
        mx = 1 + np.amax(B)
    elif len(B) == 0:
        mx = 1 + np.amax(A)
    else:    
        mx = 1 + max(np.amax(A), np.amax(B))  
    vals = np.zeros(mx)
    for index in A:
        vals[index] = 1
    for index in B:
        vals[index] = 0
    C = np.where(vals == 1)
    return C

# The function vertices2graph takes as inputs a list of vertices and an index list and returns a
# graph bipartition in the correct format for use in calculating the resulting Phi probability distributions
#
# The output format is of the form given by this example for an effect repertoire: 
# graph = [[(0,1),(1,2)],[(2,),(0,)]] means nodes 0 and 1 affect nodes 1 and 2, while node 2 affect node 0. Cause
# repetoire follows mutatis mutandis.
# The cause-effect repertoires for time-series data follow the system model and decomposition for IIT as described
# in Stephan Krohn and Dirk Ostwald, "Computing Integrated Information," Neuroscience of Consciousness, 
# Volume 2017, Issue 1, 1 January 2017, nix017, https://doi.org/10.1093/nc/nix017

def vertices2graph(vertices, index):
#    print('vertices = ', vertices, '\n')
    graph1 = [[], []]
    graph1[0] = []
    graph1[1] = []
    graph2 = [[], []]
    graph2[0] = []
    graph2[1] = []   
    for vertex in vertices:
#        print('vertex = ', vertex, '\n')
        if vertex < conf.num_of_nodes/2:
#        if vertex < num_of_nodes/2:
            graph1[0].append(vertex)
        else:
            graph1[1].append(vertex)
    graph2[0] = indexdiff(np.arange(len(index)), graph1[0])
    graph2[1] = indexdiff(np.arange(len(index)), graph1[1])
    graph = [[], []]
    graph[0] = graph1
    graph[1] = [graph2[0][0].tolist(), graph2[1][0].tolist()]
    return graph

# The function QueyranneAlgorithm implments Queyranne's algorithm for approximating the Minimum Information
# Partition (MIP). According to Jun Kitazono, Ryota Kanai, Masafumi Oizumi, "Efficient Algorithms for 
# Searching the Minimum Information Partition in Integrated Information Theory," Entropy 2018, 20(3), 173; 
# https://doi.org/10.3390/e20030173, Queyranne's Algorithm finds an approximation to the MIP, at least for
# Phi *. We use Phi 3.0 in place of Phi *, but initial experiments look promising. The code for implementing
# Queyranne's Algorithm follows the implementation from the Araya group's Practical Phi Toolbox 
# (https://figshare.com/articles/phi_toolbox_zip/3203326/10), though all probability distribution arrays have 
# been replaced with hash-tables for efficiency.
def QueyranneAlgorithm(F, index, cur_X, tuple_hash, dist_whole, args):
    if not(isinstance(index[0], list)):
#        index = np.array(index)
#        print(index)
        index = [[x] for x in index]
    indexlen = len(index)
    M = [(np.size(a)) for a in index]
    M = np.sum(M)
    indexrec = [[] for i in range(indexlen-1)]
    f = np.zeros(indexlen-2)
    for i in range(0,indexlen-2):
        pp = pendent_pair(F, index, [0], cur_X, tuple_hash, args)
        last = pp[-1]
        indexrec[i] = index[last]
        index = [index[x] for x in pp]
        graph = vertices2graph(indexrec[i], index)
#        print('Qgraph = ', graph)
        dist_part = F(graph, cur_X, tuple_hash, args)
#        print(i)
        (dist_whole_list, dist_part_list, multiplier) = wasserstein_hash_2_list(dist_whole, dist_part)
#        print('dist_whole_list = ', dist_whole_list)
#        print('dist_part_list = ', dist_part_list)
        f[i] = multiplier*wasserstein_distance(dist_part_list, dist_whole_list)
#        print('i = ', i, 'and f[i] = ', f[i])
        index[-2] = Union(index[-2], index[-1])
        index.pop()
    min_f = np.min(f)
    argmin_f = np.argmin(f)
#    print('f = ', f)
#    print('indexrec =', indexrec)
#    print('argmin_f = ', argmin_f)
    if len(indexrec[argmin_f]) > M/2:
        if isinstance(index[0], list):
            index = index[0]
        IndexOutput = indexdiff(index, indexrec[argmin_f])
        IndexOutput = IndexOutput[0]
#        print('IndexOutput in Q = ', IndexOutput, '\n')
    else:
        IndexOutput = np.sort(indexrec[argmin_f])
    return IndexOutput

# This function is the core function of the QueyranneAlgorithm
def pendent_pair(F, index, ind, cur_X, tuple_hash, args):
    ind_ind = [index[k] for k in ind]
#    print('ind_ind = ', ind_ind, '\n')
#    print('index = ', index, '\n')
#    orig_index = index
#    index = [[x] for x in index]
    for i in range(len(index) - 1):
#        index_set = set(index)
#        ind_set = set(ind)
#        indc = index_set.difference(ind_set)
        indc = indexdiff(np.arange(len(index)), ind)
        indc = indc[0]
#        print('indc = ', indc, '\n')
        candidates = [index[i] for i in indc]
#        reduced_candidates = [x for [x] in candidates]
#        reduced_ind = [x for [x] in ind_ind]
        keys = np.zeros(len(candidates))
        for j in range(len(candidates)):
#            print('index = ', index, '\n')
#            print('candidates[j] = ', candidates[j], '\n')
#           reduced_ind = [x for [x] in ind_ind]
#            print('ind = ', ind, '\n')
#            vertices_plus = Union(ind, reduced_candidates[j])
            vertices_plus = Union(ind, candidates[j])
            graph_plus = vertices2graph(vertices_plus, index)
            graph = vertices2graph(candidates[j], index)
#            print('cal_pi = ', cal_p_i(graph_plus, cur_X, tuple_hash, args))
            p_i_hash_plus = cal_p_i(graph_plus, cur_X, tuple_hash, args)
            p_i_hash = cal_p_i(graph, cur_X, tuple_hash, args)
            (p_i_list_plus, p_i_list, multiplier) = wasserstein_hash_2_list(p_i_hash_plus, p_i_hash)
            keys[j] = multiplier*wasserstein_distance(p_i_list_plus, p_i_list)
        minkey = min(keys)
#        print('minkey = ', minkey, '\n')
        minkey_ind = np.argmin(keys)
#        print('minkey_ind = ', minkey_ind, '\n')
#        print('ind = ', ind, '\n')
#        print('indc[minkey_ind] =', indc[minkey_ind], '\n')
        ind.append(indc[minkey_ind])
#        print('ind =', ind, '\n')
    return ind

# This function processes the hash-table outputs from cal_p and cal_pi core IIT functions and outputs two
# lists of reduced size (compared to full probability distributions) along with a multiplier to compensate
# for the reduction in list size. The two lists can then be inputs into the scipy stats.py wasserstein_distance
# function. In order to obtain the correct Wassrestein distance, the output from the wasserstein_distance simply
# needs to be multiplied by the multiplier
def wasserstein_hash_2_list(hash1, hash2):
    hash1_keys = list(hash1.keys())
    hash2_keys = list(hash2.keys())
    total_keys = Union(hash1_keys, hash2_keys)
    min_key = np.min(total_keys)
    max_key = np.max(total_keys)
    
    for key in total_keys:
        if key not in hash1.keys():
            hash1[key] = 0
        if key not in hash2.keys():
            hash2[key] = 0

    hash1_list = list(hash1.values())
#    print('hash1_list = ', hash1_list)
    hash2_list = list(hash2.values())
#    print('hash2_list = ', hash2_list)
    multiplier = len(hash1_list)/(max_key-min_key+1)
    return (hash1_list, hash2_list, multiplier)

# run_phi() is the main function to calculate Phi on time series data. The function loads the data, bins
# the data, according to configuration settings in phi_params, builds the hash tables representing
# the conditional cause and effect repertoire probability distributions, calculates sliding windows of Phi values
# and displays the results over time
def run_phi():
    f = open("nodes_conf.txt" , "w")
    f.write(str(conf.num_of_nodes))
    f.write("\n")
    f.close()
    
    #Get the data and bin it into the time series
    raw_time_series = load_data(conf.input_file, conf.no_of_cols_to_skip)
    #    num_of_nodes = conf.num_of_nodes
#    [raw_time_series, num_of_nodes] = compute_ica(file)
#    print(raw_time_series)
    time_series = binner(raw_time_series, conf.num_of_bins, 0, 0)
    time_series = np.array(time_series).T.tolist()
    tuple_series = tuple_time_series(time_series)

    #Calculate how many iterations are needed
    num_of_rows = len(np.array(raw_time_series).T.tolist())
    cap = num_of_rows - conf.int_len
    
    #initilize data structures
    tuple_hash = build_hash(tuple_series[:conf.int_len]) #Holds all data, build the first iteration
#    graphs = get_all_graphs() 
    phi_vals = [] #Hold phi values

    #Iterate through the sliding window and print the percent done
    for starting_value in range(0, cap ):
        print('\r', end = "")
        print("     Percent done: %2.3f" % ((float(starting_value ) * 100.00)/ (float(cap))), end =' \n')
        
        #If we need to move the window, do as such
        if not starting_value == 0:
        
            #Add the next tuple
            add_index = starting_value + conf.int_len - 1 
            cur_tuple = tuple_series[ add_index ]
            prev_tuple = tuple_series[ add_index - 1 ]
            tuple_hash = add_tuple( cur_tuple, prev_tuple, add_index, tuple_hash)
#            print('tuple_hash = ', tuple_hash)
            
            #Remove the first tuple
            remove_index = starting_value - 1
            cur_tuple = tuple_series[ remove_index ]
            nex_tuple = tuple_series[ starting_value ]
            tuple_hash = remove_tuple( cur_tuple, nex_tuple, remove_index, tuple_hash)
        
        cur_X  = tuple_series[ starting_value + conf.int_len - 3]
        
        # Compute ICA and determine the number of nodes that minimizes the sum of the square of the errors for the sliding window
        if conf.ICA_switch == True:
            [S_, num_of_nodes] = compute_ica(conf.input_file, starting_value, conf.int_len, conf.max_nodes)
            f = open("nodes_conf.txt" , "w")
            f.write(str(num_of_nodes))
            f.write("\n")
            f.close()
        
        #Initilize vectors to hold D(Pe || Pe(i)) and D(Pc || Pc(i))
        e_vals = [] 
        c_vals = []
        
        e_whole = cal_p(cur_X, tuple_hash, 0)#cal Pe(Xt|Xt-1 = X) 
#        print('e_whole = ', e_whole)
        c_whole = cal_p(cur_X, tuple_hash, 1,)#cal Pc(Xt-1|Xt = X) 
#        print('c_whole = ', c_whole)
        #iterate through the graphs
        
        ##QUEYRANNE'S ALGORITHM GOES HERE
        index = [i for i in range(conf.num_of_nodes)]
#        index = [i for i in range(num_of_nodes)]
        e_IndexOutput = QueyranneAlgorithm(cal_p_i, index, cur_X, tuple_hash, e_whole, 0)
#        print('e_IndexOutput = ', e_IndexOutput)
        c_IndexOutput = QueyranneAlgorithm(cal_p_i, index, cur_X, tuple_hash, c_whole, 1)
#        print('c_IndexOutput = ', c_IndexOutput)
        
        e_graph = vertices2graph(e_IndexOutput, index)
#        print('e_graph =', e_graph)
        c_graph = vertices2graph(c_IndexOutput, index)
#        print('c_graph =', c_graph)
        
#        for graph in graphs:
        e_part = cal_p_i(e_graph, cur_X, tuple_hash, 0)#cal Pe(i)(Xt|Xt-1 = X) 
        c_part = cal_p_i(c_graph, cur_X, tuple_hash, 1)#cal Pc(i)(Xt-1|Xt = X)
#        print('e_part = ', e_part)
#        print('c_part = ', c_part)
        
        (e_whole_list, e_part_list, multiplier) = wasserstein_hash_2_list(e_whole, e_part)
#        print('e_whole_list = ', e_whole_list)
#        print('e_part_list = ', e_part_list)
        e_val = multiplier*wasserstein_distance(e_whole_list, e_part_list)
        
        (c_whole_list, c_part_list, multiplier) = wasserstein_hash_2_list(c_whole, c_part)
#        print('c_whole_list = ', c_whole_list)
#        print('c_part_list = ', c_part_list)
        c_val = multiplier*wasserstein_distance(c_whole_list, c_part_list)
        
            #Append the values to the lists
        e_vals.append(e_val)
        c_vals.append(c_val)
        
        #Cal MIN(i \in I) for e and c    
        e_min = min(e_vals)
        c_min = min(c_vals)
#        print('e_min = ', e_min, '\n')
        
        #Add min( PHIe, PHIc) to phi values
        phi_vals.append(min(e_min, c_min))
#        print('phi_vals = ', phi_vals, '\n')
    return phi_vals


#    phi_vals = run_phi()
#    phi_series[i] = phi_vals
#    phi_mean[i] = np.mean(np.array(phi_vals))
#    phi_sd[i] = np.std(np.array(phi_vals))
#    info_string = "Phi (STD %2.6f, MEAN: %1.3f)" % (phi_sd[i], phi_mean[i])
#    plt.plot(phi_vals, label = info_string)
#    plt.legend()
#    plt.show(block = False)
#    plt.savefig("image_results/window" + str(conf.int_len) + "_noOfBins" + str(conf.num_of_bins) + ".png")
#    plt.clf()

#with open("/Users/moikle_admin/Research/SingularityNET/Python/Phi Pipeline/phi_vals.py" , "wb") as f:
#    pickle.dump(phi_vals, f)
#with open("/Users/moikle_admin/Research/SingularityNET/Python/Phi Pipeline/num_of_nodes.py" , "wb") as f:
#    pickle.dump(conf.num_of_nodes, f)

def run():
    parser = argparse.ArgumentParser(
        "Takes as input a time series of values (importance or excitation values for example) and returns a time series of estimated Tononi Phi values.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input-url", type=str,
        help="URL pointing to a CSV file.")
    parser.add_argument("--input-file", type=str, default="input.csv",
        help="The name of the input CSV file.")
    parser.add_argument("--text-output-file", type=str, default="/tmp/output.txt",
        help="The name of a file to write a textual version of the output.")
    parser.add_argument("--output-file", type=str, default="output.png", 
        help="The name of a PNG file where output will be written.")
    parser.add_argument("--timeout", type=int, default=1,
        help="Timeout in minutes.")
    parser.add_argument("--window-length", type=int, default=20,
        help="This is the desired length of the sliding window used. " + \
             "The code calculates new Phi values across each window, allowing one to see how Phi varies over time.")
    parser.add_argument("--bins", type=int, default=5,
        help="Bins the time-series data for each node into a given number of bins.")
    parser.add_argument("--nodes", type=int, default=3,
        help="initializes the number of nodes/features used to calculate Phi. Example graph " + \
             "partitions can be viewed in the files. If Independent Component Analysis (ICA) " + \
             "is turned on, this number will be recomputed on the fly, based upon the " + \
             "dimensional reduction from sk-learn's ICA routine.")
    parser.add_argument("--columns-to-skip", type=int, default=0,
        help="Number of columns to skip.")
    parser.add_argument("--csv-separator", type=str, default=",",
        help="Column separator in the CSV file.")
    parser.add_argument("--window-start", type=int, default=1,
        help="Sliding window start.")
    parser.add_argument("--max-nodes", type=int, default=50,
        help="Max nodes.")
    parser.add_argument("--ica-switch", type=bool, default=True,
        help="ICA switch.")

    args = parser.parse_args()
    if args.input_url:
        print(f"Input url: {args.input_url}")
        conf.input_file = "/tmp/input.csv"
        #data = urllib.request.urlopen(args.input_url).read(1000000).split("\n")
        url_file = urllib.request.urlopen(args.input_url)
        data = url_file.read(1000000)
        data = data.splitlines()
        with open(conf.input_file, "w") as file:
            for line in data:
                file.write(line.decode())
                file.write("\n")
    else:
        conf.input_file = args.input_file

    output_file_name = args.output_file
    text_output_file_name = args.text_output_file
    timeout_time = args.timeout

    conf.int_len = args.window_length
    conf.num_of_bins = args.bins
    conf.num_of_nodes = args.nodes
    conf.no_of_cols_to_skip = args.columns_to_skip
    conf.delim = args.csv_separator
    conf.sliding_window_start = args.window_start
    conf.max_nodes = args.max_nodes
    conf.ICA_switch = args.ica_switch

    timeout = False
    stopwatch_start = time.perf_counter()
    while not timeout:
        try:
            phi_vals = run_phi()
            break
        except LinAlgError as exception:
            print("numpy.linalg.LinAlgError: Singular matrix")
            if (time.perf_counter() - stopwatch_start) // 60 >= timeout_time:
                timeout = True
                print("Timeout")
            else:
                print("Retrying...")

    if timeout:
        shutil.copyfile(f"{os.getenv('PROJECT_DIR')}/error.png", output_file_name)
    else:
        phi_mean = np.mean(np.array(phi_vals))
        phi_sd = np.std(np.array(phi_vals))
        info_string = "Phi (STD %2.6f, MEAN: %1.3f)" % (phi_sd, phi_mean)
        with open(text_output_file_name, "w") as f:
            f.write(str(phi_mean) + "\n")
            f.write(str(phi_sd) + "\n")
            values = [str(val) for val in phi_vals]
            f.write(",".join(values) + "\n")
        #Plot the phi values
        plt.plot(phi_vals, label = info_string)
        plt.legend()
        plt.show(block = False)
        #plt.savefig("image_results/window" + str(conf.int_len) + "_noOfBins" + str(conf.num_of_bins) + ".png")
        plt.savefig(output_file_name)
        plt.clf()

if __name__ == "__main__":
    run()
