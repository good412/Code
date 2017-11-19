import copy
import build_cknn_graphs
import evaluate
import time

class TTNG_MFR:
    
    ########### Step 1: fusion the weighted ###########
    def DEMFR(self, graph_list, num_ranks, kNN, retri_amount):
        weight_merge = {}
        var_merge = 1e100
        # merge all graphs
        for i in range(0, num_ranks):
            initial_graph = copy.deepcopy(graph_list[i])
            all_keys = initial_graph.keys()
            weight_sum = {}
            for cur_key in all_keys:
                if cur_key == -1:
                    continue
                cur_weights = initial_graph[cur_key]
                for weight in cur_weights:
                    if cur_key not in weight_sum.keys():
                        weight_sum[cur_key] = weight[1]
                    else:
                        weight_sum[cur_key] += weight[1]
            for vectex in graph_list[-1]:
                if vectex == -1: continue
                if vectex not in weight_sum.keys():
                    weight_sum[vectex] = 0
            avr = 0.0
            cnt = 0
            num = 3
            for vertex in sorted(weight_sum, key=weight_sum.get, reverse=True):
                w = weight_sum[vertex]
                avr = avr + w
                cnt = cnt+1
                if cnt > num:
                    break
            avr = avr/cnt
            for vertex in weight_sum.keys():
                weight_sum[vertex] = weight_sum[vertex]/avr
            avr = 1.0
            var = 0.0
            cnt = 0
            for vertex in sorted(weight_sum, key=weight_sum.get, reverse=True):
                w = weight_sum[vertex]
                var = var+(w-avr)*(w-avr)
                cnt = cnt+1
                if cnt > num:
                    break
            var = var/cnt + 0.2
            K = var_merge/(var+var_merge)
            var_merge = var_merge*var/(var_merge+var)
                        
            for vertex in sorted(weight_sum, key=weight_sum.get, reverse=True):
                if vertex not in weight_merge.keys():
                    weight_merge[vertex] = 0.0
            for vertex in weight_merge.keys():
                if vertex not in weight_sum.keys():
                    weight_sum[vertex] = 0.0
                w1 = weight_merge[vertex]
                w2 = weight_sum[vertex]
                w = w1 + (w2-w1)*K
                weight_merge[vertex] = w
        
        selected_images = [];
        for vertex in sorted(weight_merge, key=weight_sum.get, reverse=True):
            selected_images.append(vertex)
        if len(selected_images) < retri_amount:
            voc_candidate = graph_list[0] # 0: hsv or 1000d. 1: voc
            voc_candidate = voc_candidate[-1]
            for i in voc_candidate:
                if i not in selected_images:
                    selected_images.append(i)
                    weight_merge[i] = 0
        return weight_merge, selected_images
    
    ########### Step 2: find the maximum expectation rank ###########
    def Expectation_Rank(self, vectexS, graphs, fn_fusion_result, kNN, retri_amount, weights):
        fd_stdin_fusion = open(fn_fusion_result, 'w')
        rerank_result = {}
        for node in vectexS:
            rank = []
            weight_sum = {}
            weight_sum[node] = 0
            while len(rank) < retri_amount+2:
                for vertex in sorted(weight_sum, key=weight_sum.get, reverse=True):
                    if vertex not in rank:
                        rank.append(vertex)
                        del weight_sum[vertex]
                        for v in graphs[vertex]:
                            if v in rank:
                                continue
                            # if v not in graphs[node]: continue
                            if v not in weight_sum:
                                weight_sum[v] = 0
                            weight_sum[v] += weights[vertex][v]
                            if vertex in graphs[v]:
                                weight_sum[v] += weights[v][vertex]
                        break
            rerank_result[node] = rank
                
        for i in range(len(vectexS)):
            # print i, len(rerank_result), len(rerank_result[i])
            selected_images = rerank_result[i]
            selected_images = selected_images[0: retri_amount];
            fd_stdin_fusion.write(str(i) + '.jpg ')
            for img_id in selected_images:
                fd_stdin_fusion.write(str(img_id) + ' ')
            fd_stdin_fusion.write('\n')
        fd_stdin_fusion.close()
                
#########################################################
    
if __name__ == "__main__":

    fn_fusion_result = 'data/ukbench_graph_fusion_results.txt'
    
    fn_fusion_result = 'data/ukbench_graph_fusion_results.txt'
    
    fn_fusion_reranking_result = 'data/ukbench_graph_fusion_rerank_results.txt'
    fn_fusion_reranking = 'data/ukbench_rerank_graph_fusion.txt'
    fn_label = 'data/ukbench_list_images_labels.txt'
    
    graphfusion = TTNG_MFR()
    
    fn_results = []
    fn_results.append('data/ukbench_rank_voc.txt')
    fn_results.append('data/ukbench_rank_msd.txt')
    #fn_results.append('data/ukbench_rank_cdh.txt')
    fn_results.append('data/ukbench_rank_hsv.txt')
    fn_results.append('data/ukbench_rank_cnn.txt')
    #fn_results.append('data/ukbench_rank_lbp.txt')
    fn_results.append('data/ukbench_rank_bow.txt')
    
    fn_result_rerankings = []
    fn_result_rerankings.append('data/ukbench_rerank_voc.txt')
    fn_result_rerankings.append('data/ukbench_rerank_msd.txt')
    #fn_result_rerankings.append('data/ukbench_rerank_cdh.txt')
    fn_result_rerankings.append('data/ukbench_rerank_hsv.txt')
    fn_result_rerankings.append('data/ukbench_rerank_cnn.txt')
    #fn_result_rerankings.append('data/ukbench_rerank_lbp.txt')
    fn_result_rerankings.append('data/ukbench_rerank_bow.txt')

    num_ranks = len(fn_results) # in this case, just 2 types of ranks, i.e., voc and hsv
    retri_amount = 6
    kNN = 4
    
    graph_list = []
    weights = []
    count = 1
    graphs = []
    result_length = 10200
    graph_lists = []
    vectexS = []
    print '######################  Graph Build Step  ##########################'
    
    T = time.time()
    
    for line in range(result_length):
        graphs.append([]);
        weights.append({});
    for i in range(num_ranks):
        fn_result = fn_results[i]
        fn_result_reranking = fn_result_rerankings[i]
    
    for i in range(num_ranks):
        fn_result = fn_results[i]
        fn_result_reranking = fn_result_rerankings[i]
        search_region = 4
        lam = 1 
        graph_lists.append(build_cknn_graphs.BuildKNNGraphs(fn_result, fn_result_reranking, fn_label, kNN, retri_amount, kNN + 2, lam))
    print 'Graph Build Step Time is ', time.time() - T, 's'
    print '######################  Graph Fusion Step  ##########################'
    T = time.time()
    for i in range(result_length):
        graph_list = [] 
        for j in range(num_ranks):
            graph_list.append(graph_lists[j][i])
            
        graph_list_copy = copy.deepcopy(graph_list)
        weight, graph = graphfusion.DEMFR(graph_list_copy, num_ranks, kNN, retri_amount)
        #print graph, weight
        weights[graph[0]] = weight
        graphs[graph[0]] = graph
        vectexS.append(graph[0])
    print 'Graph Fusion Step Time is ', time.time() - T, 's'
    
    print '######################  Re-Rank Step  ##########################'
    T = time.time()
    graphfusion.Expectation_Rank(vectexS, graphs, fn_fusion_result, kNN, retri_amount, weights)
    print 'Re-Rank Step; Each retrieval image time is ', (time.time() - T)/result_length*1000/retri_amount, 'ms'
    
    print "After graphs fusion:"
    evaluate.Evaluate(fn_label, fn_fusion_result, 4)

