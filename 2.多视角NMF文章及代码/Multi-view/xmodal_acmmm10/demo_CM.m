clear all;
close all;
addpath('code/');


% load ACM-MM '10 features
load 'wikipedia_info/raw_features.mat';
cat.tr=dlmread('wikipedia_info/trainset_txt_img_cat.list','\t',0,2);
cat.te=dlmread('wikipedia_info/testset_txt_img_cat.list','\t',0,2);

fprintf('----------------------------------------------\n');
fprintf(' CCA image-text\n');
% run CCA
[I.tr T.tr Mi,Mt,test] = cca3(I_tr,T_tr,I_te,T_te);
I.te=test.Xcca;
T.te=test.Ycca;
fprintf(' done.\n');

% cross-modal retrieval options
opt.metric='NC';
opt.rm=0;
COMPS=5;
fprintf(' Using %d-d CCA subspace for retrieval.\n\n',COMPS);

fprintf('----------------------------------------------\n');
fprintf(' Image queries (retrieve texts)\n');
% image queries for text retrieval
[Q,C,im2txt] = retrieval(I.te(:,1:COMPS),cat.te,T.te(:,1:COMPS),cat.te,opt);
im2txt

fprintf('\n\n----------------------------------------------\n');
fprintf(' Text queries (retrieve images)\n');
% text queries for image retrieval
[Q,C,txt2im] = retrieval(T.te(:,1:COMPS),cat.te,I.te(:,1:COMPS),cat.te,opt);
txt2im

fprintf(' done.\n');
