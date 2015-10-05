clear all
close all
clc

rng(1);

%generative process
alpha=5;
% K=10; 
K=2;
N=50;
a=1; b=1;
% pi=drchrnd(alpha*ones(1,K),1)';
pi=[0.4 0.6]';
% lambda=gamrnd(a,1/b,K,1);
lambda=[2 6]';
z=mnrnd(1,pi,N);
y=gamrnd(1,1./(z*lambda));

rng(1);
tic;
[Elambda, Epi, gam_a, gam_b]=posterior_VBfiniteMixture(y,2,100);
toc;
tic;
[lambda, pi]=posterior_finiteMixture(y,K,10000);
toc


%plot the posteriors
% figure; plot(pi(1,:),pi(2,:),'x')
figure; hist(pi(1,:),40)
% figure ; plot(lambda(1,:), lambda(2,:), 'x')
burnin=1000;
hist2d(lambda(:,burnin:end),30,30,[0 20],[0,20])