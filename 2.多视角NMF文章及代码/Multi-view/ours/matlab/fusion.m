function Y = fusion( miu, iter_num, varargin )
%FUSION 此处显示有关此函数的摘要
%% 初始化  
fusion_num = numel(varargin);
if fusion_num == 0, return; end
[n, ~] = size(varargin{1,1});
W = zeros(n,n);
for i = 1:fusion_num
    X = varargin{1,i};
    X = convert3(X);
%     sd = sqrt(mean(X.*X,2));
%     sd = sqrt(sd*sd');
    di = dist(X, X');
    sd = std(di(:));
    W = W + di/sd/fusion_num;
end
sd = std(W(:));
W = exp(-1.0*W/sd);
W = W - diag(diag(W));
D = diag(sum(W, 2));
L = D-W;
Y = inv(L+eye(n)*miu);

end

function S=convert3(X)
b=sum(X,2);
c=size(X,2);
D=repmat(b, [1,c]);
S=X./D;
end
