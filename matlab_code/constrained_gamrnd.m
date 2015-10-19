function lambda = constrained_gamrnd(gam_a, gam_b, extremes)
% gam_a=full(gam_a); gam_b=full(gam_b);
K=length(gam_a); u=rand(K,1);
endPoints=gammainc(gam_b*extremes,repmat(gam_a,1,2));
% endPoints2=gammainc(gam_b(2).*extremes,gam_a(2));
normConst = diff(endPoints,[],2);
% lambda_const2 = diff(endPoints2);
lambda = gammaincinv(endPoints(:,1)+u.*normConst,gam_a)./gam_b;
% lambda22 = gammaincinv(endPoints2(1)+u(2)*lambda_const2,gam_a(2))./gam_b(2);

%sample from the prior
idx = gam_a ==0;
if sum(idx)
    endPoints=log(extremes);
    normConst=diff(endPoints);
    lambda(idx)=exp(endPoints(1)+u(idx).*normConst);
end