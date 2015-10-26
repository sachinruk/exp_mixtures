function [lambda1_chain, lambda2_chain, state_transition] = ...
                    posteriorRjmcmc(y,  K,  extremes,  iterations)
alpha = 5;
N = length(y);

normC = diff(log(extremes));

% allocate space for lambda1/2 chains and state transitions
lambda1_chain = zeros(iterations*2, 1);
lambda2_chain = zeros(iterations*2, 2);
state_transition = zeros(iterations*2+1, 1);

%randomly generate very first iteration as state 1
state = 1; 
idx1 = 1; idx2 = 1; %index keepers
lambda1 = extremes(1)+(extremes(2)-extremes(1))*rand;
state_transition(1) = state;
lambda1_chain(idx1) = lambda1;

idx1 = idx1 +1;
l=2;
for i =1:iterations
    % jump proposals from current state to new state along with new lambdas
    if state == 1  % q(2 to 1)
        lambda1 = lambda1_chain(idx1-1);
        mu = rand(2,1);
        lambda2 = [lambda1*mu(1)/(1-mu(1)),  lambda1*(1-mu(1))/mu(1)];
        pi_12 = [mu(2), 1-mu(2)];
    else % state 2 (q 1 to 2)
        lambda2 = lambda2_chain(idx2-1,:);
        lambda1 = sqrt(prod(lambda2));
        mu(1) = lambda1/(lambda1+lambda2(2));
    end
    log_lik = logsumexp(bsxfun(@plus,-(y*lambda2),log(lambda2.*pi_12)), 2);
    log_joint_lik2 = sum(log_lik)-2*log(normC)-sum(log(lambda2))...
                     +gammaln(2*alpha)-2*gammaln(alpha)+(alpha-1)*sum(log(pi_12))-log(K);
    log_joint_lik1 = N*log(lambda1)-sum(lambda1*y)-log(normC)-log(lambda1)...
                    -log(K);
    logq = log(2)+log(lambda1)-log(mu(1))-log(1-mu(1));
    alpha_ratio = log_joint_lik2-log_joint_lik1+logq;
    A = min(0, alpha_ratio);
    if state == 2
        A = min(0, -alpha_ratio);
    end

    if A > log(rand)  % accept move
        if state == 2
            state = 1;  % switch states
            lambda1_chain(idx1) = lambda1;
            % Gibbs step
            lambda1 = q_lambda(y, extremes); %gibbs step
            lambda1_chain(idx1+1) = lambda1; 
            idx1 =idx1+2;
        else  % state 1
            state = 2;  % switch states
            lambda2_chain(idx2,:) = lambda2;
            [lambda2, pi_12]=gibbs_sampler2(y, pi_12, lambda2, alpha, extremes);
            lambda2_chain(idx2+1,:) = lambda2;
            idx2 = idx2+2;
        end
    else  % if rejected proposal, keep old value
        if state == 2
            lambda2_chain(idx2,:) = lambda2;
            [lambda2, pi_12]=gibbs_sampler2(y, pi_12, lambda2, alpha, extremes);
            lambda2_chain(idx2+1,:) = lambda2;
            idx2 = idx2+2;
        else
            lambda1_chain(idx1) = lambda1;
            lambda1 = q_lambda(y, extremes); %gibbs step
            lambda1_chain(idx1+1) = lambda1;
            idx1=idx1+2;
        end
    end
    state_transition(l:(l+1)) = [state state];
    l = l + 2;
end
lambda1_chain=lambda1_chain(1:(idx1-1));
lambda2_chain=lambda2_chain(1:(idx2-1),:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Posterior functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function z= q_z2(y, pi, lambda)
p_z = bsxfun(@plus,-y*lambda,log(pi.*lambda));
p_z = normalise(p_z);
z = mnrnd(1,p_z);

function lambda2=q_lambda2(y, z, n_k, extremes)
gam_a = n_k; gam_b = sum(bsxfun(@times,z,y));
u = rand(1,length(n_k));
endPoints1=gammainc(gam_b(1).*extremes,gam_a(1));
endPoints2=gammainc(gam_b(2).*extremes,gam_a(2));
lambda_const1 = diff(endPoints1);
lambda_const2 = diff(endPoints2);
lambda12 = gammaincinv(endPoints1(1)+u(1)*lambda_const1,gam_a(1))./gam_b(1);
lambda22 = gammaincinv(endPoints2(1)+u(2)*lambda_const2,gam_a(2))./gam_b(2);
lambda2 = [lambda12 lambda22];
idx = gam_a == 0;
if sum(idx)  % if any values with gam_a==0
    endPoints = log(extremes);
    normC = diff(endPoints);
    lambda2(idx) = exp(u(idx)*normC+endPoints(1));
end

function pi=q_pi2(n_k, alpha)
dir_par = alpha+n_k;
pi= drchrnd(dir_par, 1);

function lambda=q_lambda(y, extremes)
gam_a = length(y); gam_b = sum(y);
u = rand;
endPoints = gammainc(gam_b.*extremes,gam_a);
lambda_const = diff(endPoints);
lambda = gammaincinv(endPoints(1)+u*lambda_const,gam_a)/gam_b;

function [lambda2, pi_12]=gibbs_sampler2(y, pi_12, lambda2, alpha, extremes)
z = q_z2(y, pi_12, lambda2);
n_k = sum(z);
lambda2 = q_lambda2(y, z, n_k, extremes);
pi_12 = q_pi2(n_k, alpha);