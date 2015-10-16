function x=jeffreysPrior(iter, extremes)

normC = diff(log(extremes));
u = rand(iter,1);
x = exp(u*normC+log(extremes(1)));