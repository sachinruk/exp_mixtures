clear all
clc
close all

alpha = 0.1;
K = 2;
N = 100;
a = 1;
b = 1;
iterations=1000;
burnin=iterations*0.1;

% true generative model
rng(1);
pi = [0.4, 0.6];
lambda_ = [2, 6];
z = mnrnd(1, pi, N);
y = gamrnd(1, 1./(z*lambda_'),N, 1);
extremes = [min(1./y), max(1./y)];

%DP MCMC scheme to find posteriors
[z,lambda] = DPposterior(y, extremes, iterations, alpha);
[phi_z, Elambda]=DPVB(y,200);

c=zeros(iterations,1);
for i=1:iterations
    c(i)=length(unique(z(:,i)));
end
disp(max(c))