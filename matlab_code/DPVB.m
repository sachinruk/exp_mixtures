function [phi_z_best, Elambda_best, class_best]=...
                                            DPVB(y,iterations,varargin)
N=length(y);
T=min(N,200);

prior_flag=true;
if nargin>2
    Ealpha=varargin{1};
    prior_flag=false;
    a=0;
    b=0;
end
    

% Elambda=zeros(iterations,T);
max_lb=-inf;
restarts=20;
for j=1:restarts
    phi_z=rand(N,T);
    phi_z=bsxfun(@rdivide,phi_z,sum(phi_z,2));
    lb=zeros(iterations,1);
    if prior_flag
        Ealpha=rand;
    end
    for i=1:iterations
        [Elambda, Elnlambda, lambda_a, lambda_b]=q_lambda(y,phi_z);
        [lnV, ln1_V,V_a,V_b]=qV(phi_z,Ealpha);
        if prior_flag
            [Ealpha, a,b]=q_alpha(ln1_V);     
        end
        phi_z=qz(y,lnV,ln1_V,Elambda, Elnlambda);
        lb(i)=lnlb(y,V_a,V_b,lnV,ln1_V,...
                    lambda_a,lambda_b,Elambda,Elnlambda,phi_z,...
                    Ealpha,a,b,prior_flag);    
    end
    if lb(end)>max_lb 
        %save parameters
        max_lb=lb(end);
        phi_z_best=phi_z;
        Elambda_best=Elambda;
    end
%     [~,class]=max(phi_z,[],2);
%     subplot(restarts/5,5,j); plot(lb); xlabel(length(unique(class)));
end
[~,class_best]=max(phi_z_best,[],2);

function [Elambda,Elnlambda,gam_a,gam_b]=q_lambda(y,phi_z)
delta=1e-6;
gam_a=sum(phi_z)+delta;
gam_b=sum(bsxfun(@times,y,phi_z))+delta;
Elambda=gam_a./gam_b;
Elnlambda=psi(gam_a)-log(gam_b);


function [Ealpha, a,b]=q_alpha(ln1_V)
a=length(ln1_V)+1e-6;
b=1e-6-sum(ln1_V);
Ealpha=a/b;
% Elnalpha=psi(a)-log(b);

%beta distribution approximation
function [lnV, ln1_V, gamma_1,gamma_2]=qV(phi_z,Ealpha)
gamma_1=(sum(phi_z)+1)';
phi_z=1-cumsum(phi_z,2);
% phi_z=bsxfun(@minus,phi_z(:,end),phi_z);
gamma_2=(sum(phi_z)+Ealpha)';
% gamma=[gamma_1 gamma_2];

%expectations
lnV=psi(gamma_1)-psi(gamma_1+gamma_2);
ln1_V=psi(gamma_2)-psi(gamma_1+gamma_2);

%multinomail approxmiation
function p=qz(y,lnV,ln1_V,Elambda, Elnlambda)
ln1_V=cumsum(ln1_V);
ln1_V=[0; ln1_V(1:(end-1))];

ln_py=bsxfun(@plus,Elnlambda,-y*Elambda);
p=bsxfun(@plus,ln_py,(ln1_V+lnV)');
p=exp(bsxfun(@minus,p,max(p,[],2)));
p=bsxfun(@rdivide, p,sum(p,2));