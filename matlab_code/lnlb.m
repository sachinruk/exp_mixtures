%lower bound to likelihood
function [L]=lnlb(x,V_a,V_b,lnV,ln1_V,...
                lambda_a,lambda_b,Elambda,Elnlambda,phi_z,Ealpha,a,b)
% % a=length(ln1_V)+delta;
% % b=delta-sum(ln1_V);
delta=1e-6;
lnpalpha=-delta*Ealpha;
lnqalpha=a*log(b)-gammaln(a)-b*Ealpha;
lb_alpha=lnpalpha-lnqalpha;

occ_clusters=(sum(phi_z)>0); %occupied clusters

gamma_1=(sum(phi_z))';
phi_z2=cumsum(phi_z,2);
phi_z2=1-phi_z2;
gamma_2=sum(phi_z2)';
lnpV=gamma_1.*lnV+gamma_2.*ln1_V+(Ealpha-1)*ln1_V;
lnqV=gammaln(V_a+V_b)-(gammaln(V_a)+gammaln(V_b))+(V_a-1).*lnV+(V_b-1).*ln1_V;
lb_V=lnpV-lnqV;
% lb_V=sum(lb_V(occ_clusters));
lb_V=sum(lb_V);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prior lambda(1,1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lnp_lambda=-delta*Elambda+(delta-1)*Elnlambda+delta*log(delta)-gammaln(delta);
lnq_lambda=lambda_a.*log(lambda_b)-gammaln(lambda_a)+(lambda_a-1).*Elnlambda...
         -lambda_b.*Elambda;
lb_lambda=lnp_lambda-lnq_lambda;
lb_lambda=sum(lb_lambda(occ_clusters));

%%%%%%%%%%%%%%%%%%%%%%%%%%
% z_n
%%%%%%%%%%%%%%%%%%%%%%%%
lnqz=phi_z.*log(phi_z);
lnqz=sum(lnqz(phi_z>0));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  p(x_n|z_n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lnpx=bsxfun(@plus,-x*Elambda,Elnlambda).*phi_z;
lnpx=sum(lnpx(:));
% lb_z=-sum(lb_z);

L=lnpx+lb_lambda+lb_V+lb_alpha-lnqz;
% lik=[L,lnpx,lb_alpha,lb_V,lb_lb_mu,lb_kappa,lnqz];