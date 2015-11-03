function [z,lambdas]=DPposterior(y,extremes, iter,alpha) 

N=length(y);
% K=min(round(N*log(alpha)),5); %intial number of components
K=5; %intial number of components
z=zeros(N,iter+1);
lambdas=zeros(iter,K); %max(K,N*log(alpha))
p=repmat(1/K,K,1);
z(:,1)=randsample(1:K,N,true,p);

z_current = sparse(1:N,z(:,1),1); %convert z to a indicator matrix
n_k = full(sum(z_current,1)); % sum of each component
for i=1:iter
    % Crop n_k=0s
    idx=n_k>0;
    K=sum(idx);
    n_k=n_k(idx);
    K_old=size(lambdas,2);
    lambdas=lambdas(:,idx(1:K_old));
    z_current=z_current(:,idx);
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    %posterior on lambdas
    %%%%%%%%%%%%%%%%%%%%%%%
    gam_b = full(sum(bsxfun(@times,z_current,y)));
    lambda = constrained_gamrnd(n_k, gam_b,extremes);
    if size(lambdas,2)<length(lambda)
        extra_cols=length(lambda)-size(lambdas,2);
        lambdas=[lambdas zeros(iter,extra_cols)]; %append extra columns
    end    
    lambdas(i,:)=lambda;
        
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
        p=normalise(log_py(j,:)+log_pz); % posterior prob on z
        component=randsample(1:(K+1),1,true,p);
        z_current(j,:) = 0; %erase current assignment
        z_current(j,component)= 1; % get new assignment
        n_k = full(sum(z_current,1)); % sum of each component
        
        if component==(K+1)
            K=K+1;
            lambdaNew=jeffreysPrior(1,extremes);
            log_py=[log_py log(lambdaNew)-y*lambdaNew];
        end
    end
    [idx,j]=find(z_current);
    z(idx,i+1)=j; 
%     disp(i);
    if mod(i,100)==0
        disp(i);
    end
end
    