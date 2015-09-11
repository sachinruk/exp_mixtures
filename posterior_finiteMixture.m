function [p_z,gam_a,gam_b,dir_par]=posterior_finiteMixture(y,K,iter)

N=length(y);
a=1e-6; b=1e-6; alpha=5;
lambda=gamrnd(1,1,K,1);
pi=drchrnd(alpha*ones(1,K),1)';

for i=1:iter
    %z variable
    p_z=bsxfun(@times,exp(-y*lambda'),(pi.*lambda)');
    p_z=bsxfun(@rdivide,p_z,sum(p_z,2));
    z=mnrnd(1,p_z,N);
    %lambda variable
    n_k=sum(z);
    gam_a=a+n_k; gam_b=b+sum(bsxfun(@times,z,y));
    lambda=gamrnd(gam_a,gam_b)';
    %pi variable
    dir_par=alpha+n_k;
    pi=drchrnd(dir_par,1)';
end