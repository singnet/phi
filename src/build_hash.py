def build_hash(tuple_series):

    #Check to make sure the time series has produced correct values
    if not type(tuple_series) == list:
        print("build_hash recieved bad values")
        return None
    if not type(tuple_series[0]) == tuple:
        print("build_hash recieved bad values")
        return None
    if not len(tuple_series[0]) > 1:
        print("build_hash recieved bad values")
        return None
    if not len(tuple_series) > 1:
        print("build_hash recieved bad values")
        return None
    info_hash = {}
    i = 0
    for cur_tuple in tuple_series:
        cur_hash_list = None
        
        if not cur_tuple in info_hash:
            cur_hash_list = [[], {}, {}]
            info_hash[cur_tuple] = cur_hash_list
            
        else:
            cur_hash_list = info_hash[cur_tuple]
         
        cur_hash_list[0].append(i)
         
        try:
            nex_tuple = tuple_series[i + 1]
            if not (nex_tuple in info_hash[cur_tuple][1]):
               info_hash[cur_tuple][1][nex_tuple] = 1
            else:
               info_hash[cur_tuple][1][nex_tuple] = info_hash[cur_tuple][1][nex_tuple] + 1
        except IndexError:
            #some comment to satisfy except 
            print('')
          
        if not i ==0:  
           prev_tuple = tuple_series[i - 1]
           if not prev_tuple in info_hash[cur_tuple][2]:
               info_hash[cur_tuple][2][prev_tuple] = 1
           else:
               info_hash[cur_tuple][2][prev_tuple] = info_hash[cur_tuple][2][prev_tuple] + 1
        i = i + 1    
    return info_hash
 
"""This method is for adding a new row from the time series into the tuple_hash. Current tuple
is the tuple that is to be added, prev tuple is the tuple before the current tuple, index is
the index of the current tuple and tuple hash is the information about all tuples.""" 
def add_tuple(cur_tuple, prev_tuple, index, tuple_hash):

    #If the current tuple is not in the tuple hash, add it with default values
    if not cur_tuple in tuple_hash:
        tuple_hash[cur_tuple] = [[index], {}, {prev_tuple:1}]
        
    #other wise, add the index to the list, and update the prevous has for cur tuple
    else:
        tuple_hash[cur_tuple][0].append(index)
        pre_hash = tuple_hash[cur_tuple][2]
        if not prev_tuple in pre_hash:
            pre_hash[prev_tuple] = 1
        else:
            pre_hash[prev_tuple] = pre_hash[prev_tuple] + 1
        tuple_hash[cur_tuple][2] = pre_hash
    
    
    post_hash = tuple_hash[prev_tuple][1]
    if not cur_tuple in post_hash:
        post_hash[cur_tuple] = 1
    else:
        post_hash[cur_tuple] = post_hash[cur_tuple] + 1
    tuple_hash[prev_tuple][1] = post_hash
    return tuple_hash
    
def remove_tuple(cur_tuple, nex_tuple, index, tuple_hash):
    cur_list = tuple_hash[cur_tuple]
    #print("current" +str(cur_tuple) + " : " + str(cur_list))
    cur_list[0].remove(index)
    
    if len(cur_list[0]) == 0:
        del tuple_hash[cur_tuple]
    else:
        if cur_list[1][nex_tuple] == 1:
            del cur_list[1][nex_tuple]
        else:
            cur_list[1][nex_tuple] =  cur_list[1][nex_tuple] - 1
        tuple_hash[cur_tuple] = cur_list
    nex_list = tuple_hash[nex_tuple]
    #print("next  "+ str(nex_tuple) + " : " + str(nex_list))
    
    if nex_list[2][cur_tuple] == 1:
        del nex_list[2][cur_tuple]
    else:
        nex_list[2][cur_tuple] = nex_list[2][cur_tuple] -1
    tuple_hash[nex_tuple] = nex_list
    return tuple_hash
