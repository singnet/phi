import numpy as np
import phi_params_27Apr22 as conf
import nodes_conf as nodes_conf


n = conf.num_of_bins
N = conf.num_of_nodes
"""
Makes an index based on the tuple value
INPUT: tuple
OUTPUT: int in range 0 to num_of_bins**3 -1
"""
def get_i_fr_tuple(t):
    n = conf.num_of_bins
    N = nodes_conf.num_of_nodes
    i = 0
    for j in range(N):
        i += (t[j] - 1) * n ** (N-j-1)
#    return ((t[0] -1)*(n**2)) + ((t[1] -1)*n) + (t[2] -1)
    return i

"""
Makes a new list of zeros to hold probs
OUTPUT: list of len num_of_bins**3
"""  
def get_new_distro_list():
    return [0] * (conf.num_of_bins ** 3)

def get_new_num():
#    return [conf.num_of_bins ** (conf.num_of_nodes - 1)] * (conf.num_of_bins)
    return [0] * (conf.num_of_bins)

"""
Calculates Pe/c(X{t_or_t-1}|X{t_or_t-1}) = X)
INPUTS: the current state, or X
        the system inforamtion or tuple hash
        args-  a parameter for e or c (args == 0 means e; args == 1 means c)
OUTPUTS:A probabilty distrobution with slots for every possible state (some zero)
"""    

def hash_outer_product(dist1, dist2):
    dist1 = hash2list(dist1)
    dist2 = hash2list(dist2)
    dist3 = {}
    rows = np.nonzero(dist1)[0]
    cols = np.nonzero(dist2)[0]
    key = {}
    n1 = len(dist1)
    n2 = len(dist2)
    row_col_pairs = get_row_col_pairs(rows, cols, n1, n2)
    for item in row_col_pairs:
        key[item] = n * item[0] + item[1]
    for item in row_col_pairs:
        val1 = get_count(dist1, item[0])
#        print(val1)
        val2 = get_count(dist2, item[1])
        dist3[key[item]] = val1*val2+val1+val2
#    print('dist3 = ', dist3)    
    return dist3

def get_row_col_pairs(rows, cols, n1, n2):
    row_col_pairs = set()
    key = []
    for row in rows:
        for i in range(n1):
            row_col_pairs.add((row, i))
    for col in cols:
        for i in range(n2):
            row_col_pairs.add((i, col))
    return row_col_pairs

def get_count(dist, location):
    return dist[location]

def hash2list(hsh):
    dist = np.zeros(n)
#    print(n**node)
    for i in range(n):
        if i in hsh.keys():
            dist[i] = hsh[i]
    return dist       



def cal_p(state_tuple, tuple_hash, argument):

	n = conf.num_of_bins
	N = nodes_conf.num_of_nodes
	"""
	Create a count vector providing the probability counts (above the base number) for each node/bin combination
"""

	count = {}

	#Pe(Xt|Xt-1=X) calculation
	if argument == 0:
	
	    #Initilize varibles
		tuple_t1 = tuple_hash[state_tuple] #Get the list of information
		tuple_t = tuple_t1[1] #Grab the post information hash
#		print(tuple_t)
		tuple_t_key = list(tuple_t.keys()) #Make a list containg all the # of keys
		tuple_t_val = list(tuple_t.values()) #Make a list containg all the # of values
		prob_total = sum(tuple_t_val) #Get  the total occurnces
		prob = {}
		for node in range(0, N):   
			prob_dict = {}
			for value in range(n):
				value_count = 0
				for key in range(len(tuple_t_key)):
					if tuple_t_key[key][node] == value + 1:
						value_count += tuple_t_val[key]
				if value in prob_dict.keys():
					prob_dict[value] += value_count
				elif value_count > 0:
					prob_dict[value] = value_count
			for value in prob_dict.keys():
				prob_dict[value] = prob_dict[value]/prob_total
			prob[node] = prob_dict
#			print('prob = ', prob)
		prob_dist_whole = {}
		for key in tuple_t.keys():
#			print(key)
			key2 = get_i_fr_tuple(key)
#			print(key2)
			prob_dist_whole[key2] = 1
			for node in range(N):
				prob_dist_whole[key2] *= prob[node][key[node] - 1]                        
		return prob_dist_whole

	#Pc(Xt-1|Xt=X)  calculation	
	elif argument == 1:
	
	    #Initilize varibles
		tuple_t1 = tuple_hash[state_tuple] #Get the list of information
		tuple_t = tuple_t1[2] #Grab the post information hash
		tuple_t_key = list(tuple_t.keys()) #Make a list containg all the # of keys
		tuple_t_val = list(tuple_t.values()) #Make a list containg all the # of values
		prob_total = sum(tuple_t_val) #Get  the total occurnces
		prob = {}
		for node in range(0, N):   
			prob_dict = {}
			for value in range(n):
				value_count = 0
				for key in range(len(tuple_t_key)):
					if tuple_t_key[key][node] == value + 1:
						value_count += tuple_t_val[key]
				if value in prob_dict.keys():
					prob_dict[value] += value_count
				elif value_count > 0:
					prob_dict[value] = value_count
			for value in prob_dict.keys():
				prob_dict[value] = prob_dict[value]/prob_total
			prob[node] = prob_dict
#			print(prob)
		prob_dist_whole = {}
		for key in tuple_t.keys():
#			print(key)
			key2 = get_i_fr_tuple(key)
#			print(key2)
			prob_dist_whole[key2] = 1
			for node in range(N):
				prob_dist_whole[key2] *= prob[node][key[node] - 1]                        
		return prob_dist_whole
	
	#There should be no other value than 0 or one passed to this	
	else:
	    raise ValueError("Bad value sent in args for cal_p")
	    

def cal_p_i(graph, state_tuple, tuple_hash, args):

    #Get the information about the current graph
    partition1 = graph[0]
    partition2 = graph[1]
    
    #Setup parameters
    hash_index = None
    if args == 0: #Cal e
        hash_index = 1
        p1t1_index_tuple = partition1[0]
        p1t_index_tuple = partition1[1]
        p2t1_index_tuple = partition2[0]
        p2t_index_tuple = partition2[1]
    else: #Cal c
        hash_index = 2
        p1t1_index_tuple = partition1[1]
        p1t_index_tuple = partition1[0]
        p2t1_index_tuple = partition2[1]
        p2t_index_tuple = partition2[0]

    #Initialize count values for bi-partitions
#    pre_count_vals = np.empty([2])
#    pre_count_vals[0] = conf.num_of_bins ** (conf.num_of_nodes - len(p2t_index_tuple))
#    pre_count_vals[1] = conf.num_of_bins ** (conf.num_of_nodes - len(p1t_index_tuple))

    #Initialize counts for each node/bin
#    pre_count = np.empty([conf.num_of_nodes, conf.num_of_bins])
#    for node in p2t_index_tuple:
#        for bin in range(0, conf.num_of_bins):
#            pre_count[node][bin] = pre_count_vals[0]
#    for node in p1t_index_tuple:
#        for bin in range(0, conf.num_of_bins):
#            pre_count[node][bin] = pre_count_vals[1]
    
    
    #Find the matching values
    p1_match_hash = {}        
    p2_match_hash = {}    
    #Iterate through the hash table  
    for comp_tuple, tup_list in tuple_hash.items():
        #Check to see if the current tuple matches the state tuple on all indices required
        for index in p1t1_index_tuple:
            if not comp_tuple[index] == state_tuple[index]:
                break
        else:
            p1_match_hash = join_hash(p1_match_hash, tup_list[hash_index])
            
        for index in p2t1_index_tuple:
            if not comp_tuple[index] == state_tuple[index]:
                break
        else:
            p2_match_hash = join_hash(p2_match_hash, tuple_hash[comp_tuple][hash_index])
    for match in p1_match_hash.keys():
        key1 = get_i_fr_tuple(match)
    for match in p2_match_hash.keys():
        key2 = get_i_fr_tuple(match)
    
    final_probs = []
    for grab_index in range(0,N):
        if grab_index in p1t_index_tuple:
            final_probs.append(get_probs_from_hash(grab_index, p1_match_hash))             
        else:
            final_probs.append(get_probs_from_hash(grab_index, p2_match_hash)) 
#    print('final_probs = ', final_probs)
    distro_vals = get_new_distro_list()
    dist_num = [[] for _ in range(N)]
    dist = [[] for _ in range(N)]
    for i in range(N):
        dist_num[i] = get_new_num()
#    b_dist_num = get_new_num()
#    c_dist_num = get_new_num()
#    d_dist_num = get_new_num()
#    init = distro_vals

    for i in range(N):
        for key, prob in final_probs[i].items():
            dist_num[i][key - 1] += prob
#    for b_key, b in final_probs[1].items():
#        b_dist_num[b_key - 1] += b
#    for c_key, c in final_probs[2].items():
#        c_dist_num[c_key - 1] += c
#    for d_key, d in final_probs[3].items():
#        d_dist_num[d_key - 1] += d
#                new_prob = a*b*c
#                a_dist_num[a_key] += a
#    for a_key, a in final_probs[0].items():
#        for b_key, b in final_probs[1].items():
#            for c_key, c in final_probs[2].items():
#                index = get_i_fr_tuple(tuple((a_key,b_key,c_key)))
#                distro_vals[index] = 
#    tot_a = sum(a_dist_num)
    sum_dist = np.zeros(N)
    for i in range(N):
#        print(sum(dist_num[i]))
        sum_dist[i] = sum(dist_num[i])
        
#    b_dist = np.divide(b_dist_num, sum(b_dist_num))
#    c_dist = np.divide(c_dist_num, sum(c_dist_num))
#    d_dist = np.divide(d_dist_num, sum(d_dist_num))

    
    distro_vals = {}
    for key in p2_match_hash.keys():
#        print(key)
        key2 = get_i_fr_tuple(key)
#        print(key2)
        distro_vals[key2] = 1
        for node in range(N):
            distro_vals[key2] *= dist_num[node][key[node]-1]/sum_dist[node]                            
    return distro_vals
    
def join_hash(hash_1, hash_2):

    
    
	for key,val in hash_2.items():
		if not key in hash_1:
			hash_1[key] =val 
		else:
			hash_1[key] += val
	return hash_1
    
def get_probs_from_hash(place, match_hash):
    my_hash = {}
    
    for cur_tup, cur_val in match_hash.items():
        place_num = cur_tup[place]
        cur_val = match_hash[cur_tup]
        if not place_num in my_hash:
            my_hash[place_num] = cur_val
        else:
            my_hash[place_num] = my_hash[place_num] + cur_val
    return my_hash
