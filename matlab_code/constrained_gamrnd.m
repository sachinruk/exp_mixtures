function lambda = constrained_gamrnd(gam_a, gam_b, extremes)
K=length(gam_a); u=rand(K,1);
endPoints=gammainc(gam_b'*extremes,repmat(gam_a',1,2));
normConst = diff(endPoints,[],2);
lambda = gammaincinv(endPoints(:,1)+u.*normConst,gam_a')./gam_b';

%sample from the prior
idx = gam_a ==0;
prior_samples=sum(idx);
if prior_samples
    lambda(idx)=jeffreysPrior(prior_samples,extremes);
end