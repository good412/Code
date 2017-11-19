
def load_data(fn_result):

    print "Load data", fn_result
    fd_stdin = open(fn_result)
    img_name = []
    result_idx = []
    for line in fd_stdin:
        line = line.rstrip()
        line = line.split()
        img_name.append(line[0])
        result_idx.append(map(int, line[1:]))

    result_length = len(result_idx)
    fd_stdin.close()
    
    return img_name, result_idx, result_length

############################## Reciprocal neighbors ###################

def find_reciprocal_neighbors(img_name, result_idx, result_length, fn_result_reranking, search_region, kNN, retri_amount, lam):
    
    fd_stdin_result = open(fn_result_reranking, 'w')

    result_graphs = []
    for i in range(0, result_length):
        result_graphs.append({})
    for i in range(result_length):
        result_graph = {}

        qualified_list = []
        true_label = result_idx[i][0]
        qualified_list.append(true_label)
        result_graph[result_idx[i][0]] = [[result_idx[i][0], 1.0]]
        for j in range(search_region):
            cur_id = result_idx[i][j+1]
            cur_id_kNN = result_idx[cur_id][1:kNN]
            if result_idx[i][0] in cur_id_kNN:
                qualified_list.append(cur_id)
                (result_graph[result_idx[i][0]]).append([cur_id, 1.0]) # To keep the biggest for the weight of between i and i 
                result_graph[cur_id] = [[result_idx[i][0], 1.0]]
        for j in range(1,len(qualified_list)):
            cur_id = qualified_list[j]
            cur_id_kNN = result_idx[cur_id][0:kNN]
            common_set = set(cur_id_kNN) & set(qualified_list)
            if len(common_set) > 0:
                weight = 0.5*lam
                for k in range(1,len(cur_id_kNN)):
                    if cur_id_kNN[k] not in result_graph.keys():
                        result_graph[cur_id_kNN[k]] = [[cur_id, weight]]
                    else:
                        result_graph[cur_id_kNN[k]].append([cur_id, weight])
                    if cur_id_kNN[k] not in qualified_list:
                        qualified_list.append(cur_id_kNN[k])
        
        if len(qualified_list) <= retri_amount:
            for j in range(1, retri_amount):
                cur_id = result_idx[i][j]
                if cur_id not in qualified_list:
                    qualified_list.append(cur_id)
                    if -1 not in result_graph.keys():
                        result_graph[-1] = [cur_id]
                    else:
                        result_graph[-1].append(cur_id)
        # save the reranking results, same format as the input
        qualified_list = qualified_list[0:retri_amount]
        fd_stdin_result.write(img_name[i] + ' ')
        for cur_id in qualified_list:
            fd_stdin_result.write(str(cur_id) + ' ')
        fd_stdin_result.write('\n')
        result_graphs[i] = result_graph

    fd_stdin_result.close()
    return result_graphs

def BuildKNNGraphs(fn_result, fn_result_reranking, fn_label, kNN, retri_amount, search_region = 6, lam = 1):
    img_name, result_idx, result_length = load_data(fn_result)
    result_graphs = find_reciprocal_neighbors(img_name, result_idx, result_length, fn_result_reranking, search_region, kNN, retri_amount, lam)
    return result_graphs
