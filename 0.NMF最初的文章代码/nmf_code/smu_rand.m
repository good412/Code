%smu_rand
clear
smu=1;
m=6;
r=3;
N=200;
eps=1e-9;
err_rat=1e-3;

rand('state',1000)
W=rand(m,r);
rand('state',2000)
H=rand(r,N);
V=W*H;

num=1;
for rand_num=1:num
    rand('state',rand_num); 
    W0=rand(m,r); % randomly initialize basis
    W0=W0*diag(1./sum(W0,1));
    rand('state',rand_num*10); 
    H0=rand(r,N);
    for expnum=1:1:1
        fname=main(smu,rand_num,expnum,V,r,eps,W0,H0,err_rat)
        load(fname)
        plot(F_obj)
    end
end



