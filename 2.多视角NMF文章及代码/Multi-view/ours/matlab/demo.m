% % % data = importdata('F:\MATLAB\multi-view\ours\data\webkb\Wisconsin\wisconsin_content.mtx');
% % % feature = [];
% % % for i = 1:size(data,1)
% % %     from = data(i,1);
% % %     to = data(i,2);
% % %     value = data(i,3);
% % %     feature(from, to) = value;
% % % end
% % % 
% % % create_rank_list(feature, 'data/wisconsin_rank_content.txt')

% % labels = importdata('F:\MATLAB\multi-view\ours\data\webkb\Wisconsin\wisconsin_act.txt');
% % for i = 1:length(labels)
% %     label = labels(i);
% %     dlmwrite('data/wisconsin_list_data_labels.txt', [num2str(i) '.dat wisconsin_' num2str(label)], '-append', 'delimiter','');
% % end
% 
% labels = importdata('F:\MATLAB\multi-view\ours\data\webkb\Wisconsin\wisconsin_act.txt');
% 
% [n, ~] = size(labels);
% labels_real = zeros(1,n);
% rlabels = {[], [], [], [], []};
% for i = 1:n
%     label = labels(i);
%     rlabels{label} = [rlabels{label}, i];
%     labels_real(i) = label;
% end
% 
% save('data/wisconsin', 'G', 'n', 'nClass', 'labels_real', 'rlabels');

data = importdata('vectors.bin');
feature = [];
for i = 1:size(data,1)
    feature(data(i,1)+1,:) = data(i,2:end);
end
nmf_feature = convert3(feature);
[n, G, nClass, labels, ~] = load_file('animal');
baseline_spectral(nmf_feature,nClass,labels');

% % 
% labels = importdata('F:\MATLAB\multi-view\ours\data\Cora\cora_act.txt');
% % for i = 1:2708
% %     label = labels(i);
% %     dlmwrite('data/cora_list_data_labels.txt', [num2str(i) '.dat cora_' num2str(label)], '-append', 'delimiter','');
% % end


% % % % data = importdata('F:\MATLAB\multi-view\ours\data\3sources\3sources.disjoint.clist');
% % % % data = data.data;
% % % % labels = zeros(1, 416);
% % % % for i = 1:6
% % % %     for j = 1:103
% % % %         if isnan(data(i,j)), continue; end;
% % % %         labels(data(i,j)) = i;
% % % %     end
% % % % end
% % % % labels(1) = 1;
% % % % labels(5) = 2;
% % % % labels(42) = 3
% % % % labels(4) = 4;
% % % % labels(6) = 5;
% % % % labels(2) = 6;
% % % % for i = 1:416
% % % %     label = labels(i);
% % % %     dlmwrite('data/reuter_list_data_labels.txt', [num2str(i) '.dat digit_' num2str(label)], '-append', 'delimiter','');
% % % % end


% 
% data = importdata('data/cora_graph_fusion_results.txt');
% data = data.data;
% 
% [n, ~] = size(data);
% m = 20;
% nClass = 7;
% G = zeros(n,n);
% for from = 1:n
%     for toInd = 1:m
%         to = data(from,toInd)+1;
%         G(from,to) = 1;
%     end
% end
% labels_real = zeros(1,n);
% rlabels = {[], [], [], [], [], [], []};
% for i = 1:n
%     label = labels(i);
%     rlabels{label} = [rlabels{label}, i];
%     labels_real(i) = label;
% end
% 
% save('data/cora', 'G', 'n', 'nClass', 'labels_real', 'rlabels');
