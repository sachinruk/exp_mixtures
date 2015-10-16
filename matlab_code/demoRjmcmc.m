clear all
clc
close all

alpha = 5;
K = 2;
N = 100;
a = 1;
b = 1;
iterations=20000;
burnin=iterations*0.1;

rng(1);
pi = [0.4, 0.6];
lambda_ = [2, 6];
z = mnrnd(1, pi, N);
y = gamrnd(1, 1./(z*lambda_'),N, 1);
extremes = [min(1./y), max(1./y)];

[lambda1, lambda2, states] = posteriorRjmcmc(y, K, extremes, iterations);

states=states(burnin:end);
state1 = sum(states == 1);
state2 = sum(states == 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% likelihood
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iter = 1e6;
lambda12 = jeffreysPrior(iter, extremes);
lambda22 = jeffreysPrior(iter, extremes);
pi = betarnd(alpha, alpha, iter, 1);
log_py = zeros(iter, 1);
normC = diff(log(extremes));

for i= 1:iter
    a = log(pi(i))+log(lambda12(i))-lambda12(i)*y;
    b = log(1-pi(i))+log(lambda22(i))-lambda22(i)*y;
    log_pyi = [a b];
    log_py(i) = sum(logsumexp(log_pyi, 2));
end

% a = log(pi)+log(lambda12)-lambda12*y.T
% b = log(1-pi)+log(lambda22)-lambda22*y.T

log_py_k2 = logsumexp(log_py, 1)-log(iter);
log_py_k1 = -log(normC)-N*log(sum(y))+gammaln(N)+log(diff(gammainc(sum(y)*extremes,N)));

p_k1 = 1./(1.+exp(log_py_k2-log_py_k1));
disp(p_k1);
disp(state1/(state1+state2));
% plt.figure()
% plt.plot(lambda1)
% plt.show()
% plt.figure()
% plt.plot(lambda2[:, 0], lambda2[:, 1], '.')
% plt.show()

