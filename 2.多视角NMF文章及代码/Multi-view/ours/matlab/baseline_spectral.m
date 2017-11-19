function baseline_spectral(X,numClust,truth)
% INPUT:
% X: N x P data matrix. Each row is an example
% numClust: desired number of clusters
% truth: N x 1 vector of ground truth clusterings
% OUTPUT:
% C, U, F, P, R: clustering, U matrix, F-score, Precision, Recall
    if (min(truth)==0)
        truth = truth + 1;
    end
    fprintf('running k-means...\n');
    nmii = [];
    aci = [];
    for i=1:20
        C = kmeans(X,numClust,'EmptyAction','drop','Distance','sqeuclidean', 'Replicates',10, 'Start', 'plus');
        nmii(i) = nmi(C, truth);
        aci(i) = CalcMetrics(C, truth);
        fprintf('nmi: mean=%f\n', nmii(i));
        fprintf('ac: mean=%f\n', aci(i));
%         nmii(i) = compute_nmi(truth,C);
%         [Fi(i),Pi(i),Ri(i)] = compute_f(truth,C);
%         [ARi(i),RIi(i),MIi(i),HIi(i)]=RandIndex(truth,C);
    end
   fprintf('nmi: mean=%f(%f), max=%f\n', mean(nmii), std(nmii), max(nmii));
   fprintf('ac: mean=%f(%f), max=%f\n', mean(aci), std(aci), max(aci));
end