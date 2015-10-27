function log_joint_lik2=logJointLik(y,lambda,pi,K, normC)
dim=length(lambda);
log_lik = logsumexp(bsxfun(@plus,-(y*lambda),log(lambda.*pi)), 2);
log_joint_lik2 = sum(log_lik)-dim*log(normC)-sum(log(lambda))-log(K);