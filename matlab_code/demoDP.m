clear all
clc
close all

alpha = 0.1;
K = 2;
% N = 50;
N=1000;
a = 1;
b = 1;
iterations=1e3;
burnin=iterations*0.1;

% true generative model
rng(1);
pi = [0.4, 0.6];
% lambda_ = [2, 5];
lambda_ = [2, 10];
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
% disp(max(c))

figure; plot(c); axis([1 iterations 0 10])
figure; hist(c)

z=class_true;
nmi_mcmc=zeros(size(z_inf,2),1);
ari_mcmc=zeros(size(z_inf,2),1);
for i=1:size(z_inf,2)
    nmi_mcmc(i)=nmi(z,z_inf(:,i));
end
nmi_vb=nmi(z,class_vb);
% ari(z,z_inf(:,end))
for i=1:size(z_inf,2)
    ari_mcmc(i)=ari(z,z_inf(:,i));
end
ari_vb=ari(z,class_vb);

figure; plot(nmi_mcmc); hold on; plot([1 iterations],[nmi_vb nmi_vb],'r')
figure; plot(ari_mcmc); hold on; plot([1 iterations],[ari_vb ari_vb],'r')

% figure; hist(lambda(lambda(:,2)>0,2),30)