function [z,lambda]=DPposterior(y,extremes, iter,alpha) 

N=length(y);
K=min(round(N*log(alpha)),5); %intial number of components
z=zeros(N,iter+1);t
% lambda=zeros(iter,K); %max(K,N*log(alpha))
p=repmat(1/K,K,1);
z(:,1)=randsample(1:K,N,true,p);

z_current = sparse(1:N,z(:,1),1); %convert z to a indicator matrix
n_k = full(sum(z_current,1)); % sum of each component
for i=1:iter
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    %posterior on lambdas
    %%%%%%%%%%%%%%%%%%%%%%%
    gam_b = full(sum(bsxfun(@times,z_current,y)));
    lambda = constrained_gamrnd(n_k', gam_b',extremes);
    lambda = [lambda' jeffreysPrior(1,extremes)];
    
    %%%%%%%%%%%%%%%%%%%%%%
    %posterior on z
    %%%%%%%%%%%%%%%%%%%%%%%
    % log p(y|z,lambda)
    log_py = bsxfun(@plus,log(lambda),-y*lambda); % a NxK matrix
    for j=1:N
        n_k_without_zi=n_k-z_current(j,:);
        % log p(z|alpha)
        log_pz=log([n_k_without_zi alpha]);
        p=normalise(log_py(i,:)+log_pz); % posterior prob on z
        component=randsample(1:(K+1),1,true,p);
        z_current(j,:) = 0; %erase current assignment
        z_current(j,component)= 1; % get new assignment
        n_k = full(sum(z_current,1)); % sum of each component
    end
    [idx,j]=find(s);
    z(idx,i+1)=j; 
end
    