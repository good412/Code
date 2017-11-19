clear all;
close all;
addpath('code/');
addpath('3rd-party/liblinear/matlab/');

% load ACM-MM '10 features
load 'wikipedia_info/raw_features.mat';
cat.tr=dlmread('wikipedia_info/trainset_txt_img_cat.list','\t',0,2);
cat.te=dlmread('wikipedia_info/testset_txt_img_cat.list','\t',0,2);


fprintf('----------------------------------------------\n');    
fprintf(' Learning Semantic mappings\n');                                   

% train Log. Regression
% try some cross-validation procedure 
% for the model parameters

% model for Image
fprintf('- image\n');
model_I = train(cat.tr, sparse(I_tr),'-s 0 -B 1 -c 30 -q');
%ops.verb=2;model_I = lin_train(cat.tr, sparse(I_tr),ops);
[predicted_label, acc, smn_I.te] = predict(cat.te, sparse(I_te), model_I,' -b 1'); 
% model for Text
fprintf('- text\n');
model_T = train(cat.tr, sparse(T_tr),'-s 0 -B 1 -c 30 -q');
%ops.verb=2;model_T = lin_train(cat.tr, sparse(T_tr),ops);
[predicted_label, acc, smn_T.te] = predict(cat.te, sparse(T_te), model_T,' -b 1'); 

fprintf('done.\n');


% cross-modal retrieval options
opt.metric='NC';
opt.rm=1;

fprintf('----------------------------------------------\n'); 
fprintf(' Image queries (retrieve texts)\n');                
% image queries for text retrieval
[Q,C,im2txt] = retrieval(smn_I.te,cat.te,smn_T.te,cat.te,opt);
im2txt

fprintf('\n\n----------------------------------------------\n');  
fprintf(' Text queries (retrieve images)\n');                     
% text queries for image retrieval
[Q,C,txt2im] = retrieval(smn_T.te,cat.te,smn_I.te,cat.te,opt);
txt2im


fprintf(' done.\n'); 

