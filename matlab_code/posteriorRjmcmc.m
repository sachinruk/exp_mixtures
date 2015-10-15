function [lambda1_chain, lambda2_chain, state_transition] = ...
                    posteriorRjmcmc(y,  K,  extremes,  iterations)
    alpha = 5;
    N = len(y);
%     pi = zeros(iterations+1, K);
%     pi(1) = random.dirichlet(alpha*ones(1, K)(1), 1)

    normC = diff(log(extremes));
    state = 1;

%     lambda_chain = list()
    lambda1 = extremes(1)+(extremes(2)-extremes(1))*rand;
    lambda1_chain = zeros(iterations*2, 1);
    lambda2_chain = zeros(iterations*2, 2);

    state_transition = zeros(iterations*2+1, 1);
    state_transition(1) = state;
%     l = 1

    idx1 = 1;
    idx2 = 1;
    lambda1_chain(idx1) = lambda1;
    idx1 = idx +1;
    for i =1:iterations
        if state == 1
            lambda1 = lambda1_chain(idx1-1);
            mu = rand(2,1);
            lambda2 = [lambda1*mu(1)/(1-mu(1)),  lambda1*(1-mu(1))/mu(1)];
            pi_12 = [mu(2), 1-mu(2)];
        else % state 2
            lambda2 = lambda2_chain(idx2-1);
            lambda1 = prod(lambda2);
            mu(1) = lambda1/(lambda1+lambda2(2));
        end
        log_lik = logsumexp(bsxfun(@plus,-(y*lambda2'),log(lambda2.*pi_12)), 2);
        log_joint_lik2 = sum(log_lik)-2*log(normC)-sum(log(lambda2))...
                         -log(K);
        log_joint_lik1 = N*log(lambda1)-sum(lambda1*y)-log(normC)-log(lambda1)...
                        -log(K);
        logq = log(2)+log(lambda1)-log(mu(1))-log(1-mu(1));
        alpha_ratio = log_joint_lik2-log_joint_lik1+logq;
        % A = min(1, exp(alpha_ratio)
        A = min(0, alpha_ratio);
        if state == 2
            % A = min(1, exp(-alpha_ratio)
            A = min(0, -alpha_ratio);
        end

        if A > log(rand)  % accept move
            if state == 2
                state = 1;  % switch states
                lambda1_chain(idx1) = lambda1;
                % new values of lambda (birth step?)
                lambda1 = q_lambda(y, extremes);
                lambda1_chain(idx1+1) = lambda1;
                idx1 =idx1+2;
            else  % state 1
                state = 2;  % switch states
                lambda2_chain(idx2, :) = lambda2;
                % new values of lambda (birth step?)
                z = q_z2(y, pi_12, lambda2);
                n_k = sum(z);
                lambda2 = q_lambda2(y, z, n_k, extremes);
                pi_12 = q_pi2(n_k, alpha);
                lambda2_chain(idx2+1, :) = lambda2;
                idx2 = idx2+2;
            end
        else  % keep old value
            if state == 2
                lambda2_chain(idx2, :) = lambda2;
                z = q_z2(y, pi_12, lambda2);
                n_k = sum(z);
                lambda2 = q_lambda2(y, z, n_k, extremes);
                pi_12 = q_pi2(n_k, alpha);
                lambda2_chain(idx2+1, :) = lambda2;
                idx2 = idx2+2;
            else
                lambda1_chain(idx1) = lambda1;
                lambda1 = q_lambda(y, extremes);
                lambda1_chain(idx1+1) = lambda1;
                idx1=idx1+2;
            end
        end
        state_transition(l:(l+2)) = [state state];
        l = l + 2;
    end