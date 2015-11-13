function density(y,n)

if nargin<2
    n=20;
end

[f,x]=hist(y,n);%# create histogram from a normal distribution.
% g=1/sqrt(2*pi)*exp(-0.5*x.^2);%# pdf of the normal distribution

%#METHOD 1: DIVIDE BY SUM
% figure(1)
% bar(x,f/sum(f));hold on
% plot(x,g,'r');hold off

%#METHOD 2: DIVIDE BY AREA
figure;
bar(x,f/trapz(x,f));hold on
% plot(x,g,'r');hold off