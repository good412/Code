function [Acc,Cls,Obj,Us,Ut,H,Vs,Vt] = GTL3(Xs,Xt,Ys,Yt,options)

% Mingsheng Long, Jianmin Wang, Guiguang Ding, Dou Shen, Qiang Yang. 
% Transfer Learning with Graph Co-Regularization.
% IEEE Transactions on Knowledge and Data Engineering (TKDE), 2013.

if nargin < 5
    error('No algorithm parameters provided!');
end
if ~isfield(options,'p')
    options.p = 10;
end
if ~isfield(options,'lambda')
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
p = options.p;
lambda = options.lambda;
gamma = options.gamma;
sigma = options.sigma;
iters = options.iters;
data = options.data;

fprintf('GTL3: data=%s  p=%d  lambda=%f  gamma=%f  sigma=%f\n',data,p,lambda,gamma,sigma);

%% Set predefined variables (Yt only for test)
X = [Xs,Xt];
Y = [Ys;Yt];
m = size(X,1);
c = length(unique(Y));
ns = size(Xs,2);
nt = size(Xt,2);
YY = [];
for i = reshape(unique(Y),1,c)
    YY = [YY,Y==i];
end
YYs = YY(1:ns,:);
YYt = YY(ns+1:end,:);

%% Data normalization (for classification)
Xs = Xs*diag(sparse(1./sqrt(sum(Xs.^2))));
Xt = Xt*diag(sparse(1./sqrt(sum(Xt.^2))));

%% Construct graph Laplacian
manifold.k = p;
manifold.Metric = 'Cosine';
manifold.NeighborMode = 'KNN';
manifold.WeightMode = 'Cosine';
manifold.bNormalizeGraph = 0;
[Wus,Dus] = laplacian(Xs,manifold);
[Wut,Dut] = laplacian(Xt,manifold);
[Wvt,Dvt] = laplacian(Xt',manifold);
% manifold.NeighborMode = 'Supervised';
% manifold.gnd = Ys;
% [Wvs,Dvs] = laplacian(Xs',manifold);

%% Initialization
Us = rand(m,c);
Ut = rand(m,c);
H = rand(c,c);
Vs = 0.1 + 0.8*YYs;
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

%% Graph Co-Regularized Transfer Learning (GTL3)
Acc = [];
Obj = [];
for it = 0:iters
    
    %% Alternating Optimization
    if it>0
        Us = Us.*sqrt((Xs*(Vs*H')+lambda*Wus*Us)./(Us*(H*(Vs'*Vs)*H')+lambda*Dus*Us+eps));
        Us = Us./(repmat(sum(Us.^2,1).^0.5,size(Us,1),1)+eps);

        Ut = Ut.*sqrt((Xt*(Vt*H')+lambda*Wut*Ut)./(Ut*(H*(Vt'*Vt)*H')+lambda*Dut*Ut+eps));
        Ut = Ut./(repmat(sum(Ut.^2,1).^0.5,size(Ut,1),1)+eps);
        
        Vs = Vs.*sqrt(Vs./(Vs*(Vs'*Vs)+eps));
        
        Vt = Vt.*sqrt((Xt'*(Ut*H)+gamma*Wvt*Vt+sigma*Vt)./(Vt*(H'*(Ut'*Ut)*H)+gamma*Dvt*Vt+sigma*Vt*(Vt'*Vt)+eps));
        
        H = H.*sqrt((Us'*(Xs*Vs)+Ut'*(Xt*Vt))./(Us'*(Us*H*(Vs'*Vs))+Ut'*(Ut*H*(Vt'*Vt))+eps));
    end
    
    %% Compute accuracy
    [~,Cls] = max(Vt,[],2);
    [~,Lbl] = max(YYt,[],2);
    acc = numel(find(Cls == Lbl))/nt;
    Acc = [Acc;acc];
    
    %% Compute objective
    O = 0;
%     % Comment for fast evaluations
%     O = norm(Xs-Us*(H*Vs'),'fro')^2 + norm(Xt-Ut*(H*Vt'),'fro')^2 ...
%         + sigma*norm(Vs'*Vs-eye(c,c),'fro')^2 + sigma*norm(Vt'*Vt-eye(c,c),'fro')^2 ...
%         + lambda*trace(Us'*(Dus-Wus)*Us) + lambda*trace(Ut'*(Dut-Wut)*Ut) ...
%         + gamma*trace(Vs'*(Dvs-Wvs)*Vs) + gamma*trace(Vt'*(Dvt-Wvt)*Vt);
    Obj = [Obj;O];
    
    if mod(it,10)==0
        fprintf('[%d]  objective=%0.10f  accuracy=%0.4f\n',it,O,acc);
    end
end

fprintf('Algorithm GTL3 terminated!!!\n\n\n');

end