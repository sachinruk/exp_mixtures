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

% true generative model
rng(1);
pi = [0.4, 0.6];
lambda_ = [2, 6];
z = mnrnd(1, pi, N);
y = gamrnd(1, 1./(z*lambda_'),N, 1);
extremes = [min(1./y), max(1./y)];

%DP MCMC scheme to find posteriors
[z,lambda] = DPposterior(y, extremes, iterations, alpha);