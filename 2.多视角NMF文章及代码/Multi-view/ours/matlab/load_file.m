function [n, G, nClass, labels, rlabels] = load_file(dataset,~)
load(['data' filesep dataset '.mat']);
labels = labels_real;

end

