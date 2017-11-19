function [U, V, centroidV, log, ac] = MultiNMF(X, K, label, options)
% This is a module of Multi-View Non-negative Matrix Factorization(MultiNMF)
%
% Notation:
% X ... a cell containing all views for the data
% K ... number of hidden factors
% label ... ground truth labels
% Written by Jialu Liu (jliu64@illinois.edu)

viewNum = length(X);
Rounds = options.rounds;

U_ = [];
V_ = [];

U = cell(1, viewNum);
V = cell(1, viewNum);

j = 0;
log = 0;
ac = 0;

% initialize basis and coefficient matrices
while j < 3
    j = j + 1;
    if j == 1
        [U{1}, V{1}] = NMF(X{1}, K, options, U_, V_);
        printResult(V{1}, label, K, options.kmeans);
    else
        [U{1}, V{1}] = NMF(X{1}, K, options, U_, V{viewNum});
        printResult(V{1}, label, K, options.kmeans);        
    end
    for i = 2:viewNum
        [U{i}, V{i}] = NMF(X{i}, K, options, U_, V{i-1});
        printResult(V{i}, label, K, options.kmeans);
    end
end

optionsForPerViewNMF = options;
oldL = 100;

tic
j = 0;
while j < Rounds
    j = j + 1;
    if j==1
        centroidV = V{1};
    else
        centroidV = options.alpha(1) * V{1};
        for i = 2:viewNum
            centroidV = centroidV + options.alpha(i) * V{i};
        end
        centroidV = centroidV / sum(options.alpha);
    end
    logL = 0;
    for i = 1:viewNum
        tmp1 = X{i} - U{i}*V{i}';
        tmp2 = V{i} - centroidV;
        logL = logL + sum(sum(tmp1.^2)) + options.alpha(i) * sum(sum(tmp2.^2));
    end
    log(end+1) = logL;
    if(oldL < logL)
        U = oldU;
        V = oldV;
        logL = oldL;
        j = j - 1;
        disp('restrart this iteration');
    else
        ac(end+1) = printResult(centroidV, label, K, options.kmeans);
    end
    
    oldU = U;
    oldV = V;
    oldL = logL;
    
    for i = 1:viewNum
        optionsForPerViewNMF.alpha = options.alpha(i);
        [U{i}, V{i}] = PerViewNMF(X{i}, K, centroidV, optionsForPerViewNMF, U{i}, V{i});
    end
end
toc