clear all;
close all;                               
addpath('code/');                        
addpath('3rd-party/liblinear/matlab/');  

% load ACM-MM '10 features
load 'wikipedia_info/raw_features.mat';
cat.tr=dlmread('wikipedia_info/trainset_txt_img_cat.list','\t',0,2);
cat.te=dlmread('wikipedia_info/testset_txt_img_cat.list','\t',0,2);

fprintf('----------------------------------------------\n');  
fprintf(' CCA image-text\n');                                 
% CM
[I.tr T.tr Mi,Mt,test] = cca3(I_tr,T_tr,I_te,T_te); 
I.te=test.Xcca;                                     
T.te=test.Ycca;                                     
fprintf(' done.\n');

fprintf('----------------------------------------------\n');   
fprintf(' Learning Semantic mappings\n');                      
                                                               
% SM on CM (i.e. SCM)
COMPS=7;
fprintf(' Using %d-d CCA subspace to obtain image semantic mappings.\n',COMPS);
model_I = train(cat.tr, sparse(I.tr(:,1:COMPS)),'-s 0 -B 1 -c 30 -q');
[predicted_label, acc, smn_I.te] = predict(cat.te, sparse(I.te(:,1:COMPS)), model_I,' -q -b 1'); 
% uncomment the following line for a different 
% number of (CCA) components on each modality
%COMPS=7;
%fprintf(' Using %d-d CCA subspace to obtain text semantic mappings.\n',COMPS);
model_T = train(cat.tr, sparse(T.tr(:,1:COMPS)),'-s 0 -B 1 -c 30 -q');
[predicted_label, acc, smn_T.te] = predict(cat.te, sparse(T.te(:,1:COMPS)), model_T,' -q -b 1'); 
fprintf(' done.\n');


% cross-modal retrieval options
opt.metric='NC';
opt.rm=0;

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
