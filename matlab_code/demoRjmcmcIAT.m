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
chains=30;
lambda1=cell(chains,1); lambda2=cell(chains,1);
for i=1:chains
    [lambda1{i}, lambda2{i}] = posteriorRjmcmc(y, K, extremes, iterations,gibbs_steps);
end

% i=3;
% window=200;
% iat(lambda1{i},window)
% iat(lambda2{i}(:,1),window)
% iat(lambda2{i}(:,2),window)

for i=1:chains
    burnin = round(length(lambda1{i})*0.1);
    lambda1{i}=lambda1{i}(burnin:end); 
    burnin = round(length(lambda2{i})*0.1);
    lambda2{i}=lambda2{i}(burnin:end,:); 
end

meanLambda1=cellfun(@mean,lambda1);
varLambda1=cellfun(@var,lambda1);
meanLambda2=zeros(chains,2); varLambda2=zeros(chains,2);
for i=1:chains
    meanLambda2(i,:)=mean(lambda2{i});
    varLambda2(i,:)=var(lambda2{i});
end
iatLambda2 = bsxfun(@rdivide,var(meanLambda2),...
                    bsxfun(@rdivide,varLambda2,cellfun(@length,lambda2)));
iatLambda1 = var(meanLambda1)./(varLambda1./cellfun(@length,lambda1));

window=100;
iatLambda1est=zeros(chains,1);
iatLambda2est=zeros(chains,2);
for i=1:chains
    for j=1:2
        iatLambda2est(i,j)=iat(lambda2{i}(:,j),window);
    end
    iatLambda1est(i)=iat(lambda1{i},window);
end
