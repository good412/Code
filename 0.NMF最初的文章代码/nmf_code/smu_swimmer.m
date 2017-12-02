
% smu_swimmer
clear
smu=2;


load swimmer_data
for i=1:256
    V(:,i)=reshape(swim(i,:,:),32*32,1);
end
% load swimmer_noise
% H0=V(:,1:16)';
% model_fig_image(H0,4,4,32,32);


[m N]=size(X_swim);
err_rat=0.025;
r=17;
eps=1e-9;
c=max(max(X_swim));
X_swim=X_swim/c;

num=1;
for rand_num=1:num
    rand('state',rand_num); 
    W0=rand(m,r); % randomly initialize basis
    W0=W0*diag(1./sum(W0,1));
    rand('state',rand_num*10); 
    H0=rand(r,N);
    for expnum=1:1:1
        fname=main(smu,rand_num,expnum,X_swim,r,eps,W0,H0,err_rat);
        load(fname)
        plot(F_obj)
    end
end
