clear all
close all
clc

rng(1);

%generative process
alpha=5;
K=10; N=1000;
a=1; b=1;
pi=drchrnd(alpha*ones(1,K),1)';
lambda=gamrnd(a,1/b,K,1);
z=mnrnd(1,pi,N);
y=gamrnd(1,1./(z*lambda));

[p_z,gam_a,gam_b,dir_par]=posterior_finiteMixture(y,K,100);