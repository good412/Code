function [W H F_obj]=NMFns(V,r,eps,maxiter,W0,H0,S,err_rat)

[n N]=size(V); % V contains your data in its column vectors
% % % rand('state',st); %best
% % % W=rand(n,r); % randomly initialize basis
% % % W=W*diag(1./sum(W,1));
% % % rand('state',st);
% % % H=rand(r,N); % randomly initialize encodings
W=W0;
H=H0;
F_obj(1,1)= sum(sum((V-W*S*H).*(V-W*S*H)))/sum(sum(V.*V));
for iter=1:maxiter
    H=H.*((W*S)'*V+eps)./((W*S)'*(W*S)*H+eps);
    W=W.*(V*(S*H)'+eps)./(W*(S*H)*(S*H)'+eps);
    W=W*diag(1./sum(W,1));
    
    F_obj(1,iter+1)= sum(sum((V-W*FI*H).*(V-W*FI*H)))/sum(sum(V.*V));
    if F_obj(iter+1)<err_rat
        break;
    end
end


