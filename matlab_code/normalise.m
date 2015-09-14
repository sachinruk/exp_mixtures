function p=normalise(logp)
%returns exp(logp)/sum(exp(logp)) without numerical problems for a NxD
%matrix
max_logp=max(logp,[],2);
logp=bsxfun(@minus,logp,max_logp);
p=exp(logp);
p=bsxfun(@rdivide,p,sum(p,2));