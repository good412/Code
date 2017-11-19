function  [nmf_feature] = graph2feature(dataset, feaLen, type)

if type == 0    %synthetic data
    [n, M, nClass, labels, ~] = load_gn(dataset);
    G=sparse(M(:,1),M(:,2),ones(n,1));
    G=full(G);
elseif type == 1  %real data
    [n, M, nClass, labels, ~] = load_real(dataset);
    G=sparse(M(:,1),M(:,2),ones(n,1));
    G=full(G);
elseif type == 2
    [n, G, nClass, labels, ~] = load_file(dataset);
end

nmf_feature = WW(G,feaLen);
baseline_spectral(nmf_feature,nClass,optSigma(nmf_feature),labels');
end

function nmf_feature = WW(G,feaLen)
%use lse nmf
X=XGNMF(G,feaLen);
nmf_feature=convert3(X);
end

function mX=XGNMF(network,feaLen)
options = [];
options.maxIter = 1000;
options.alpha = 1;
options.nRepeat = 2;
[U,V] = GNMF(network,feaLen,zeros(size(network)),options);
mX = V;
end

