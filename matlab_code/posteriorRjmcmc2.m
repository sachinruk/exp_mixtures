function [lambda_chain, pi_chain,state_transition] = ...
                posteriorRjmcmc2(y,K,extremes,iterations,gibbs_steps,dims,alpha)
normC = diff(log(extremes));

% allocate space for lambda1/2 chains and state transitions
lambda_chain=cell(dims,1);
pi_chain=cell(dims,1);
for i=1:dims
    lambda_chain{i} = zeros(iterations*(1+gibbs_steps), i);
    pi_chain{i} = zeros(iterations*(1+gibbs_steps), i);
end
state_transition = zeros(iterations*(1+gibbs_steps)+1, 1);

%randomly generate very first iteration as state 1
state = 1; goingup=true;
idx=1;
state_transition(1) = state;
lambda_chain{state}(idx) = jeffreysPrior(1,extremes);
pi_chain{state}(idx)=1;
l=2;
for i =1:iterations
%     disp(i);
    % jump proposals from current state to new state along with new lambdas
    lambdaOld = lambda_chain{state}(idx,:);
    piOld=pi_chain{state}(idx,:);
    log_joint1=logJointLik(y,lambdaOld,piOld,K, normC,alpha);
    if goingup  
%         idx2=choose_idx(1,state); %choose an index to split
        idx2=randsample(state,1);
        mu = rand(2,1);
        lambda1=lambdaOld(idx2);
        lambda2=[lambda1*mu(1)/(1-mu(1)),  lambda1*(1-mu(1))/mu(1)];        
        lambdaNew = [lambdaOld(1:(idx2-1)) lambda2 lambdaOld((idx2+1):state)];
        piNew = [piOld(1:(idx2-1)) piOld(idx2).*[mu(2), 1-mu(2)] ...
                                                    piOld((idx2+1):state)];
        log_joint2=logJointLik(y,lambdaNew,piNew,K, normC,alpha);
        log_joint_ratio=log_joint2-log_joint1+log(piOld(idx2))...
                        -log(state+1);
    else % if goingdown
%         idx2=choose_idx(2,state);
        idx2=randsample(state,2);
        lambda2 = lambdaOld(idx2);
        lambda1 = sqrt(prod(lambda2));
        mu(1) = lambda1/(lambda1+lambda2(2));        
%         lambdaNew = [lambdaOld(~idx2) lambda1];
%         lambdaNew =slideIn(lambdaOld, lambda1, idx2);
        lambdaNew=lambdaOld; 
        lambdaNew(idx2(1))=lambda1; lambdaNew(idx2(2))=[];
        piNew=piOld; piNew(idx2(1))=sum(piOld(idx2)); piNew(idx2(2))=[];
%         piNew = [piOld(~idx2) sum(piOld(idx2))];
        log_joint2=logJointLik(y,lambdaNew,piNew,K, normC,alpha);
        log_joint_ratio=log_joint1-log_joint2+log(sum(piOld(idx2)))...
                        -log(state);
    end    
    logq = 2*log(2)+log(lambda1)-log(mu(1))-log(1-mu(1));
    alpha_ratio = log_joint_ratio+logq;
    A = min(0, alpha_ratio);
    if ~goingup
        A = min(0, -alpha_ratio);
    end

    idx=idx+1;
    if A > log(rand)  % accept move
        if goingup
            state = state + 1;  % switch states
        else  % if going down
            state = state - 1;  % switch states            
        end
        lambda_chain{state}(idx,:) = lambdaNew;
        pi_chain{state}(idx,:)=piNew;
    else  % if rejected proposal, keep old value
        lambda_chain{state}(idx,:) = lambdaOld;
        pi_chain{state}(idx,:)=piOld;
        piNew=piOld; lambdaNew=lambdaOld;
    end
    for j=1:gibbs_steps
        [lambdaNew, piNew]=gibbs_sampler(y, piNew, lambdaNew, alpha, extremes);
        lambda_chain{state}(idx+j,:) = lambdaNew;
        pi_chain{state}(idx+j,:)=piNew;
    end
    idx=idx+gibbs_steps;
    goingup=nextmove(state,dims);
    state_transition(l:(l+gibbs_steps)) = [state repmat(state,1,gibbs_steps)];
    l = l + gibbs_steps+1;
end

% %fill up missing states
% for i=1:dims
%     for j=1:i
%         lambda_chain{i}(:,j)=fillLast(lambda_chain{i}(:,j));
%         pi_chain{i}(:,j)=fillLast(pi_chain{i}(:,j));
%     end
% end


function goingup=nextmove(state,dims)
if state==1
    goingup=true;
elseif state==dims
    goingup=false;
else
    if rand>0.5
        goingup=true;
    else
        goingup=false;
    end
end