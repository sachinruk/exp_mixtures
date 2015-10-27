function val=iat(y,window)
% Integrated Autocorrelation function of series y

acf=autocorr(y,window);
val=1+2*sum(acf(2:end));