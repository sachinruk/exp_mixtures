function [z,lambda]=DPposterior(y,K,iter,alpha) 

N=length(y);
z=zeros(N,K,iter);
lambda=zeros(K,iter);
p=repmat(1/K,K,1);
z(:,:,1)=mnrnd(1,p,N);

for i=1:iter
    z_current = z(:,:,iter);
    n_k = sum(z_current,1);
    gam_b = sum(bsxfun(@times,z_current,y));
    lambda = gamrnd(n_k, 1./gam_b);
    
    %posterior on z
    log_py = bsxfun(@plus,log(lambda),-y*lambda);
    for j=1:N
        n_k_without_zi=n_k-z_current(j,:);
    end
end
    