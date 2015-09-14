function [Elambda, Epi, gam_a, gam_b]=posterior_VBfiniteMixture(y,K,iter)

alpha=5;
% N=length(y);
% Elambda=zeros(K,iter+1);
%Epi=zeros(K,iter+1);
Elambda=gamrnd(1,1,K,1);
Elnlambda=log(Elambda);
Epi=drchrnd(alpha*ones(1,K),1)';
Elnpi=log(Epi);
Epi2=zeros(length(Epi),iter);
Elambda2=zeros(length(Epi),iter);

for i=1:iter
    %z variable
    Ez=q_z(y,Elambda,Elnlambda,Elnpi);
    %z=mnrnd(1,q_z);
    %lambda variable
    [Elambda, Elnlambda,gam_a,gam_b]=q_lambda(y,Ez);
    %Elambda(:,i+1)=gamrnd(gam_a,1./gam_b)';
    %Epi variable
    [Epi, Elnpi]=q_pi(Ez,alpha);
    %save variables
    Epi2(:,i)=Epi;
    Elambda2(:,i)=Elambda;
end
figure; plot(Epi2(1,:)); title('pi posterior convergence')
figure; subplot(121); plot(Elambda2(1,:))
subplot(122); plot(Elambda2(2,:)); title('lambdas posterior convergence')


function Ez=q_z(y,Elambda,Elnlambda,Elnpi)
Ez=bsxfun(@plus,-y*Elambda',(Elnpi+Elnlambda)');
Ez=normalise(Ez);

function [Elambda, Elnlambda,gam_a,gam_b]=q_lambda(y,Ez)
a=0.1; b=0.1;
n_k=sum(Ez);
gam_a=a+n_k'; gam_b=b+sum(bsxfun(@times,Ez,y))';
Elambda=gam_a./gam_b;
Elnlambda=psi(gam_a)-log(gam_b);

function [Epi, Elnpi]=q_pi(Ez,alpha)
n_k=sum(Ez);
alpha=alpha+n_k';
sum_alpha=sum(alpha);
Epi=alpha./sum_alpha;
Elnpi=psi(alpha)-psi(sum_alpha);
