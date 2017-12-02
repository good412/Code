
function [W H F_obj]=NMF(V,r,eps,maxiter,W0,H0,err_rat)
[n N]=size(V); % V contains your data in its column vectors

W=W0;
H=H0;
F_obj(1,1)=sum(sum((V-W*H).*(V-W*H)))/sum(sum(V.*V));
for iter=1:maxiter
    H=H.*(W'*V+eps)./(W'*W*H+eps);
    W=W.*(V*H'+eps)./(W*H*H'+eps);
     W=W*diag(1./sum(W,1));
    F_obj(1,iter+1)=sum(sum((V-W*H).*(V-W*H)))/sum(sum(V.*V));
     if F_obj(iter+1)<err_rat
        break;
     end
end



