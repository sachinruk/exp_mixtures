function [phi_z, Elambda]=DPVB(y,iterations)
N=length(y);
T=100;
phi_z=ones(N,T)./T;
Ealpha=rand;
Elambda=zeros(iterations,T);
for i=1:iterations
    [Elambda(i,:), Elnlambda]=q_lambda(y,phi_z);
    [lnV, ln1_V,V_a,V_b]=qV(phi_z,Ealpha);
    [Ealpha, a,b]=q_alpha(ln1_V);     
    phi_z=qz(y,lnV,ln1_V,Elambda(i,:), Elnlambda);
end

function [Elambda,Elnlambda]=q_lambda(y,phi_z)
delta=1e-9;
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
phi_z=cumsum(phi_z,2);
phi_z=bsxfun(@minus,phi_z(:,end),phi_z);
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