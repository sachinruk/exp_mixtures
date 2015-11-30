setwd('~/Documents/thesis/exp_mixtures/stan')
y=read.csv('y250.csv',header = F)
data=list(N=dim(y)[1], k=2, y=unlist(y))

library(rstan)
fit <- stan(file = 'model.stan', data = data, 
            iter = 1000, chains = 4)

la=extract(fit, permuted= TRUE)

library(gplots)
hist2d(la$lambda,nbins = 100)