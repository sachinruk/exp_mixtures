clear all
clc
close all

alpha = 1;
% K = 2;
N = 100;
a = 1;
b = 1;
iterations=2000;
burnin=iterations*0.1;

% CRP generative model
rng(1);
theta = gamrnd(1,10,1000,1); %create vector of parameters
z=zeros(N,1); z(1)=1;
for i=2:N
    z_true = sparse(1:(i-1),z(1:(i-1)),1); %convert z to a indicator matrix
    n_k = full(sum(z_true,1)); % sum of each component
    p=[n_k alpha]./(i-1+alpha);
    z(i) = randsample(1:length(p),1,true,p);
end
[~,class_true]=max(z,[],2);
z_true = sparse(1:N,z,1); %convert z to a indicator matrix
K=size(z_true,2);
y = gamrnd(1, 1./(z_true*theta(1:K)),N, 1);
extremes = [min(1./y), max(1./y)];

%DP MCMC scheme to find posteriors
[z_inf,lambda] = DPposterior(y, extremes, iterations, alpha);

%VB to find posterior
[phi_z, Elambda,class_vb]=DPVB(y,500);
unique(class_vb)


c=zeros(iterations,1);
for i=1:iterations
    c(i)=length(unique(z_inf(:,i)));
end
% disp(max(c))

figure; plot(c); axis([1 iterations 0 max(c)])
figure; hist(c);

mcmc_c=confusionmat(z,z_inf(:,end));
vb_c=confusionmat(z,class_vb);

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

disp(length(unique(z)));
figure; plot(lambda(:,1))
% z_inf2 = sparse(1:N,z_inf(:,4687),1); %convert z to a indicator matrix
% lof_z_true=lof(z_true); lof_z_inf2=lof(z_inf2);
% figure; subplot(121); imagesc(lof_z_true);
% subplot(122); imagesc(lof_z_inf2);
% 
% [idx,j]=find(lof_z_inf2);
% z_inf3=zeros(size(z));
% z_inf3(idx)=j;
% C=confusionmat(z,z_inf3);
% figure; imagesc(C)