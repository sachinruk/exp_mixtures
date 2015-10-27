function [lambda, pi]=gibbs_sampler(y, pi, lambda, alpha, extremes)
z = q_z(y, pi, lambda);
n_k = sum(z);
lambda = q_lambda(y, z, n_k, extremes);
pi = q_pi(n_k, alpha);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Posterior functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function z= q_z(y, pi, lambda)
p_z = bsxfun(@plus,-y*lambda,log(pi.*lambda));
p_z = normalise(p_z);
z = mnrnd(1,p_z);

function lambda=q_lambda(y, z, n_k, extremes)
gam_a = n_k; gam_b = sum(bsxfun(@times,z,y));
lambda = constrained_gamrnd(gam_a, gam_b, extremes)';

function pi=q_pi(n_k, alpha)
dir_par = alpha+n_k;
pi= drchrnd(dir_par, 1);