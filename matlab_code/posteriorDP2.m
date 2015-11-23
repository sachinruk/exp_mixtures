function [z,lambdas]=posteriorDP2(y,extremes, iter,alpha) 

N=length(y);
% T=min(N,200);
T=200;
z=zeros(N,iter);
v=betarnd(1,alpha,T,1)';
log_v=log(v);
log1_v=cumsum(log(1-v));
ln_pi=log_v+[0 log1_v(1:(end-1))];
pi=exp(ln_pi);
z_current=mnrnd(1,pi,N);
% pi=rand(N,T); 
% pi=bsxfun(@rdivide,pi,sum(pi,2));
% z_current=mnrnd(1,pi);
n=sum(z_current); sum_y=y'*z_current;
lambdas=zeros(T,iter);

for i=1:iter
    lambdas(:,i)=q_lambda(n,sum_y,extremes);
    [v, ln_pi]=q_v(z_current,n,alpha);
    [z_current,n,sum_y]=q_z(y,lambdas(:,i)',ln_pi);
    [z(:,i),~]=find(z_current');
end

function lambda=q_lambda(a,b,extremes)
lambda = constrained_gamrnd(a, b, extremes)';

function [v, ln_pi]=q_v(z,n,alpha)
a=n+1; b=sum(cumsum(z,2,'reverse'))-n+alpha;
v=betarnd(a,b);
log_v=log(v);
log1_v=cumsum(log(1-v));
ln_pi=log_v+[0 log1_v(1:(end-1))];

function [z_current,n,sum_y]=q_z(y,lambdas,ln_pi)
log_pi=bsxfun(@plus,log(lambdas)+ln_pi,-y*lambdas);
log_c=logsumexp(log_pi,2);
pi=exp(bsxfun(@minus,log_pi,log_c));
z_current=mnrnd(1,pi);
n=sum(z_current); sum_y=y'*z_current;
% [z,~]= find(z_current');