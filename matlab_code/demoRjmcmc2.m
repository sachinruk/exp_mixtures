clear all
% clc
close all

alpha = 1;
%K = 10;
N = 250;
a = 1;
b = 1;
iterations= 1e5;
% burnin=iterations*0.1;

% true generative model
rng(1);
pi = [0.4, 0.6];
lambda_ = [2, 6];
z = mnrnd(1, pi, N);
y = gamrnd(1, 1./(z*lambda_'),N, 1);
extremes = [min(1./y), max(1./y)];

iterations2=1; gibbs_steps=1; models=6;
state=zeros(iterations2,models);
for j=1:iterations2
    %MCMC scheme to find posteriors
    [lambda_chain, pi_chain, states] = posteriorRjmcmc2(y,extremes,...
                                          iterations,gibbs_steps,models,alpha);
    burnin=round(length(states)*0.1);
    for i=1:models
        state(j,i) = sum(states(burnin:end)== i);
    end
    % find how many are from state 1 and 2
end

% figure; plot(states)
pis=bsxfun(@rdivide,state,sum(state,2));
disp([mean(pis)-std(pis); mean(pis)+std(pis)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% exact marginal likelihood
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % p(y|k=2)
% iterations2=5;
% p_k=zeros(iterations2,models);
% for l=1:iterations2
%     log_py=zeros(iterations,models);
%     log_py_model=zeros(models,1);
%     for j=1:models
%         lambda=zeros(iterations,j);
%         for k=1:j
%             lambda(:,k)=jeffreysPrior(iterations, extremes);
%         end
%         pi = drchrnd(repmat(alpha,1,j),iterations);
%         for i= 1:iterations
%             lambdas=lambda(i,:); pis=pi(i,:);
%             log_lik = logsumexp(bsxfun(@plus,-(y*lambdas),log(lambdas.*pis)), 2);
%             log_py(i,j) = sum(log_lik);
%         end
%         log_py_model(j)=logsumexp(log_py(:,j),1)-log(iterations);
%     end
% 
%     % p_k_est=zeros(1,length(state));
%     
%     for i=1:models
%     %     p_k_est(i)=state(i)/sum(state);
%         p_k(l,i)=exp(log_py_model(i)-logsumexp(log_py_model));
%     end
% end
% % disp(p_k_est)
% disp([mean(p_k)-std(p_k); mean(p_k)+std(p_k)]);
% 
% 
% for i=1:length(lambda_chain)
%     lambdas=lambda_chain{i}(:,1);
%     idx=lambdas<50 & lambdas>0;
%     lambdas=lambdas(idx);
%     figure; hist(lambdas,100);
% end

idx=lambda_chain{2}(:,1)>0;
hist2d(lambda_chain{2}(idx,:),30,30,[0 20], [0 20]); title('lambda2 posterior')
figure; hist3(lambda_chain{2}(idx,:),[100,100])
figure; hist(pi_chain{2}(idx,1),30)

%testing against old 2-state algorithm
%[lambda_chain, pi_chain, states] = posteriorRjmcmc(y, K, extremes, iterations,gibbs_steps,alpha);
% 
% %trace plots of lambda1
% figure; plot(lambda1(burnin:end)); title('lambda_{11}')
% figure; plot(lambda2(burnin:end,1)); title('lambda_{12}')
% figure; plot(lambda2(burnin:end,2)); title('lambda_{22}')
% figure; plot(pi_chain(burnin:end)); title('\pi trace');
% 
% figure; autocorr(lambda1(burnin:end)); title('\lambda_{11}')
% figure; autocorr(lambda2(burnin:end,1)); title('\lambda_{12}')
% figure; autocorr(lambda2(burnin:end,2)); title('\lambda_{22}')
% figure; autocorr(pi_chain(burnin:end)); title('\pi');
