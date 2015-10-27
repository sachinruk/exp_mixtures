clear all
% clc
close all

alpha = 5;
K = 2;
N = 250;
a = 1;
b = 1;
iterations=50000;
% burnin=iterations*0.1;

% true generative model
rng(1);
pi = [0.4, 0.6];
lambda_ = [2, 6];
z = mnrnd(1, pi, N);
y = gamrnd(1, 1./(z*lambda_'),N, 1);
extremes = [min(1./y), max(1./y)];
gibbs_steps=1; models=3;
%MCMC scheme to find posteriors
[lambda_chain, pi_chain, states] = posteriorRjmcmc2(y,K,extremes,...
                                            iterations,gibbs_steps,models);

% find how many are from state 1 and 2
burnin=round(length(states)*0.1);
states=states(burnin:end);
state=zeros(models,1);
for i=1:models
    state(i) = sum(states == i);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% exact marginal likelihood
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% p(y|k=2)
iter = 5*1e4;
log_py=cell(models,1);
log_py_model=zeros(models,1);
for j=1:models
    lambda=zeros(iter,j);
    for k=1:j
        lambda(:,k)=jeffreysPrior(iter, extremes);
    end
    pi = drchrnd(repmat(alpha,1,j),iter);
    log_py{j} = zeros(iter, 1);
    for i= 1:iter
        lambdas=lambda(i,:); pis=pi(i,:);
        log_lik = logsumexp(bsxfun(@plus,-(y*lambdas),log(lambdas.*pis)), 2);
        log_py{j}(i) = sum(log_lik);
    end
    log_py_model(j)=logsumexp(log_py{j})-log(iter);
end
p_k2=exp(log_py_model(2)-logsumexp(log_py_model));
p_k2_est=state(2)/sum(state);

% p_k1 = 1./(1.+exp(log_py_k2-log_py_k1));
% disp(strcat('exact posterior of p(k=1|y): ',num2str(p_k1)));
% disp(strcat('simulated posterior of p(k=1|y): ',num2str(state1/(state1+state2))));
% 
% %plot the outputs of simulated lambdas
% figure()
% hist(lambda1(lambda1<50 ),100); title('lambda11 posterior')
% hist2d(lambda2,100,100,[0 20],[0 20]); title('lambda2 posterior')
% view(90,270)
% idx=lambda2(:,1)<20; figure; hist(lambda2(idx,1),200)
% idx=lambda2(:,2)<20; figure; hist(lambda2(idx,2),200)
% 
% figure; hist(pi_chain,100);
% 
% %trace plots of lambda1
% figure; plot(lambda1(burnin:end)); title('lambda_{11}')
% figure; plot(lambda2(burnin:end,1)); title('lambda_{12}')
% figure; plot(lambda2(burnin:end,2)); title('lambda_{22}')
% figure; plot(pi_chain(burnin:end)); title('\pi trace');
% 
% figure; autocorr(lambda1(burnin:end)); title('\lambda_{11}')
% figure; autocorr(lambda2(burnin:end,1)); title('\lambda_{12}')
% figure; autocorr(lambda2(burnin:end,2)); title('\lambda_{22}')
% figure; autocorr(pi_chain(burnin:end)); title('\pi');
