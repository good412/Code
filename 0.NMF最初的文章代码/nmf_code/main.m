function fname=main(smu,rand_num,expnum,V,r,eps,W0,H0,err_rat)

[m N]=size(V);
maxiter=200;
fname = ['results' num2str(smu) '/smu' num2str(smu) '_sources' num2str(r) '_rand' num2str(rand_num) '_experiment_' num2str(expnum)];    
%------------------------------------------------------------------------------
%                            Decode 'expnum'
%------------------------------------------------------------------------------

switch expnum,
    case 1,
        t0=cputime; 
       [W H F_obj]=NMF(V,r,eps,maxiter,W0,H0,err_rat);
       time=cputime-t0;
        save(fname,'W','H','F_obj','time');
   case 2,
        t0=cputime;
        theta=0.5; 
        S=(1-theta)*eye(r,r)+theta/r*ones(r,1)*ones(1,r);
        [W  H F_obj]=NMFns(V,r,eps,maxiter,W0,H0,S,err_rat);
        time=cputime-t0;
        save(fname,'W','H','F_obj','time');
end
end


