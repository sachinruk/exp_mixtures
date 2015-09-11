function [lambda, pi]=posterior_finiteMixture(y,K,iter)

% N=length(y);
a=1e-6; b=1e-6; alpha=5;
lambda=zeros(K,iter+1);
pi=zeros(K,iter+1);
lambda(:,1)=gamrnd(1,1,K,1);
pi(:,1)=drchrnd(alpha*ones(1,K),1)';


for i=1:iter
    %z variable
    p_z=bsxfun(@plus,-y*lambda(:,i)',log(pi(:,i).*lambda(:,i))');
    p_z=normalise(p_z);
    z=mnrnd(1,p_z);
    %lambda variable
    n_k=sum(z);
    gam_a=a+n_k; gam_b=b+sum(bsxfun(@times,z,y));
    lambda(:,i+1)=gamrnd(gam_a,1./gam_b)';
    %pi variable
    dir_par=alpha+n_k;
    pi(:,i+1)=drchrnd(dir_par,1)';
end