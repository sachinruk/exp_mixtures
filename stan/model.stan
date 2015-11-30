data {
	int<lower=1> N;
	int<lower=1> k;
	real<lower=0> y[N];
}
parameters {
	simplex[k] pi;
	real<lower=min(y), upper=max(y)> lambda[k];
}
model {
	real ps[k];
	lambda~exponential(1);
	for (i in 1:N){
		for (j in 1:k){
			ps[j]<-log(pi[j])+gamma_log(y[i],1,1/lambda[j]);
		}
		increment_log_prob(log_sum_exp(ps));
	}
}
