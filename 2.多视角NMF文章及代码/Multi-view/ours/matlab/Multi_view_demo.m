datasets = {'digit'};
for i = 1:length(datasets)
    lses = [];
    max_nmf_feature = [];
    type = 2;
    for feaLen = 8 : 15
        nmf_feature_tmp = graph2feature(datasets{i}, feaLen, type);
    end
end