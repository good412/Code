function [Acc,Cls,Obj,U,Vs,Vt] = GTL2(Xs,Xt,Ys,Yt,options)

% Mingsheng Long, Jianmin Wang, Guiguang Ding, Dou Shen, Qiang Yang. 
% Transfer Learning with Graph Co-Regularization.
% IEEE Transactions on Knowledge and Data Engineering (TKDE), 2013.

if nargin < 5              % nargin，Number of function input arguments
    error('No algorithm parameters provided!');
end
if ~isfield(options,'p') %Output  1 (true) if 结构 contains the field, or logical 0 (false) if not.
    options.p = 10;
end
if ~isfield(options,'lambda')  %选项中更改为默认值
    options.lambda = 0.1;
end
if ~isfield(options,'gamma')
    options.gamma = 1.0;
end
if ~isfield(options,'sigma')
    options.sigma = 10.0;
end
if ~isfield(options,'iters')
    options.iters = 100;
end
if ~isfield(options,'data')
    options.data = 'default';
end
p = options.p;                %迭代中用结构的简写
lambda = options.lambda;
gamma = options.gamma;
sigma = options.sigma;
iters = options.iters;
data = options.data;

fprintf('GTL2: data=%s  p=%d  lambda=%f  gamma=%f  sigma=%f\n',data,p,lambda,gamma,sigma);

%% Set predefined variables (Yt only for test)
Y = [Ys;Yt];                %变成一维
m = size([Xs,Xt],1);        %size(a,1)求矩阵的行数
c = length(unique(Y));      %Y聚得10类
ns = size(Xs,2);
nt = size(Xt,2);
YY = [];
for i = reshape(unique(Y),1,c)
    YY = [YY,Y==i];              %Y是1维3800行的，这个for循环只执行了i=1的一次，执行10次会成为矩阵的。又变成矩阵了，晕
end
YYs = YY(1:ns,:);               %源、目标域分真假矩阵的行向量
YYt = YY(ns+1:end,:);

%% Data normalization (for classification)  normalize data sets by X←X/||X||
Xs = Xs*diag(sparse(1./sqrt(sum(Xs.^2))));      %sum后变成256*1,sqrt为对里边的各元素进行开方
Xt = Xt*diag(sparse(1./sqrt(sum(Xt.^2))));      %这样平方和相加就为1

%% Construct graph Laplacian    
manifold.k = p;
manifold.Metric = 'Cosine';              % 余弦
manifold.NeighborMode = 'KNN';
manifold.WeightMode = 'Cosine';
manifold.bNormalizeGraph = 0;            %？？？？？
[Wus,Dus] = laplacian(Xs,manifold);      % 强大的函数help affinity ；Wus是用affinity(Xs,manifold)得到的
[Wut,Dut] = laplacian(Xt,manifold);
[Wvt,Dvt] = laplacian(Xt',manifold);
%Metric:'Cosine' 将使用两个向量的余弦值来评估它们之间的“紧密度”。一种在信息检索中使用的流行的相似性度量。
%NeighborMode - 指示如何构建图形。在两个节点之间放置一个边，当且仅当它们在彼此的k个近邻中。 您需要在选项中提供参数k。 默认k = 5。
%WeightMode: 'Cosine'如果节点i和j连接，则放置权重余弦（x_i，x_j）。 只能在“余弦”度量下使用。
% manifold.NeighborMode = 'Supervised';
% manifold.gnd = Ys;
% [Wvs,Dvs] = laplacian(Xs',manifold);

%% Initialization
U = rand(m,c);
Vs = 0.1 + 0.8*YYs;         %Ys, Vt by logistic regression trained on fXs; Ysg.
if isfield(options,'Yt0') && size(options.Yt0,1)==nt
    Vt = [];
    for i = reshape(unique(Y),1,c)
        Vt = [Vt,options.Yt0==i];
    end
    options.Yt0 = [];
    Vt = 0.1 + 0.8*Vt;
else
    Vt = rand(nt,c);
end

%% Graph Co-Regularized Transfer Learning (GTL)
Acc = [];
Obj = [];
for it = 0:iters
    
    %% Alternating Optimization
    if it>0
        U = U.*sqrt((Xs*Vs+Xt*Vt+lambda*Wus*U+lambda*Wut*U)./(U*(Vs'*Vs)+U*(Vt'*Vt)+lambda*Dus*U+lambda*Dut*U+eps));
        
        Vs = Vs.*sqrt(Vs./(Vs*(Vs'*Vs)+eps));

        Vt = Vt.*sqrt((Xt'*U+gamma*Wvt*Vt+sigma*Vt)./(Vt*(U'*U)+gamma*Dvt*Vt+sigma*Vt*(Vt'*Vt)+eps));
    end
    
    %% Compute accuracy
    [~,Cls] = max(Vt,[],2);     %创建一个矩阵并计算每一行中的最大元素。
    [~,Lbl] = max(YYt,[],2);
    acc = numel(find(Cls == Lbl))/nt;  %find: Find the nonzero elements,按列数，找元素的位置;n = numel(A) returns the number of elements
    Acc = [Acc;acc];
    
    %% Compute objective
    O = 0;
%     % Comment for fast evaluations
%     O = norm(Xs-U*Vs','fro')^2 + norm(Xt-U*Vt','fro')^2 ...
%         + sigma*norm(Vs'*Vs-eye(c,c),'fro')^2 + sigma*norm(Vt'*Vt-eye(c,c),'fro')^2 ...
%         + lambda*trace(U'*(Dus-Wus)*U) + lambda*trace(U'*(Dut-Wut)*U) ...
%         + gamma*trace(Vs'*(Dvs-Wvs)*Vs) + gamma*trace(Vt'*(Dvt-Wvt)*Vt);
    Obj = [Obj;O];
    
    if mod(it,10)==0
        fprintf('[%d]  objective=%0.10f  accuracy=%0.4f\n',it,O,acc);
    end
end

fprintf('Algorithm GTL2 terminated!!!\n\n\n');

end