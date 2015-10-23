clear all
clc
close all

alpha = 5;
K = 2;
N = 100;
a = 1;
b = 1;
iterations=20000;
% burnin=iterations*0.1;

% true generative model
rng(1);
pi = [0.4, 0.6];
lambda_ = [2, 6];
z = mnrnd(1, pi, N);
y = gamrnd(1, 1./(z*lambda_'),N, 1);
extremes = [min(1./y), max(1./y)];
gibbs_steps=3;
%MCMC scheme to find posteriors
[lambda1, lambda2, states] = posteriorRjmcmc(y, K, extremes, iterations,gibbs_steps);

% find how many are from state 1 and 2
burnin=length(states)*0.1;
states=states(burnin:end);
state1 = sum(states == 1);
state2 = sum(states == 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% exact marginal likelihood
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% p(y|k=2)
iter = 1e6;
lambda12 = jeffreysPrior(iter, extremes);
lambda22 = jeffreysPrior(iter, extremes);
pi = betarnd(alpha, alpha, iter, 1);
log_py2 = zeros(iter, 1);

for i= 1:iter
    a = log(pi(i))+log(lambda12(i))-lambda12(i)*y;
    b = log(1-pi(i))+log(lambda22(i))-lambda22(i)*y;
    log_pyi = [a b];
    log_py2(i) = sum(logsumexp(log_pyi, 2));
end

log_py_k2 = logsumexp(log_py2, 1)-log(iter);
% p(y|k=1)
normC = diff(log(extremes));
log_py_k1 = -log(normC)-N*log(sum(y))+gammaln(N)+log(diff(gammainc(sum(y)*extremes,N)));

p_k1 = 1./(1.+exp(log_py_k2-log_py_k1));
disp(strcat('exact posterior of p(k=1|y): ',num2str(p_k1)));
disp(strcat('simulated posterior of p(k=1|y): ',num2str(state1/(state1+state2))));

%plot the outputs of simulated lambdas
figure()
hist(lambda1(lambda1<50 ),100); title('lambda11 posterior')
hist2d(lambda2,40,40,[0 50],[0 50]); title('lambda2 posterior')
view(90,270)
idx=lambda2(:,1)<20; figure; hist(lambda2(idx,1),200)
idx=lambda2(:,2)<20; figure; hist(lambda2(idx,2),200)

%trace plots of lambda1
figure; plot(lambda1(burnin:end))
figure; plot(lambda2(burnin:end,1))
figure; plot(lambda2(burnin:end,2))

window=200;
iat(lambda1,window)
iat(lambda2(:,1),window)
iat(lambda2(:,2),window)