function [W,D] = laplacian(X,manifold)

W = affinity(X,manifold);   % X（即fea）: Rows of vectors of data points. Each row is x_i
                            %affinity才是真正的构建了图像拉普拉斯
if manifold.bNormalizeGraph         %chuan传进来的是0啊，还是要看W的值
    D = 1./sqrt(sum(W));
    D(isinf(D)) = 0;
    D = diag(sparse(D));
    W = D*W*D;
    D(D>0) = 1;
else
    D = diag(sparse(sum(W)));
end

end