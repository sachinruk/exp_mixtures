clear all
clc
close all

alpha = 0.1;
K = 2;
N = 50;
a = 1;
b = 1;
iterations=1e3;
burnin=iterations*0.1;

% true generative model
rng(1);
pi = [0.4, 0.6];
lambda_ = [2, 5];
z = mnrnd(1, pi, N); [~,class_true]=max(z,[],2);
y = gamrnd(1, 1./(z*lambda_'),N, 1);
extremes = [min(1./y), max(1./y)];

[phi_z, Elambda, class_vb]=DPVB(y,1000);

%DP MCMC scheme to find posteriors
[z_inf,lambda] = DPposterior(y, extremes, iterations, alpha);


c=zeros(iterations,1);
for i=1:iterations
    c(i)=length(unique(z_inf(:,i)));
end
disp(max(c))

figure; plot(c); axis([1 iterations 0 10])

confusionmat(class_true,z_inf(:,end))
confusionmat(class_true,class_vb)