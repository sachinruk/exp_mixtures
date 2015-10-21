% DPP
% ============================================================
% Generate Data from finite mixture
clear
finite=0;
if finite==1
    n=200;
    m1=1;%mean of normal distribution 1
    m2=3;%mean of normal distribution 2
    s1=0.5;%std of normal distribution 1
    s2=0.4;%std of normal distribution 2
    pie_temp=0.5;
    u=rand(n,1);
    y=normrnd(m1,s1,n,1);
    y2=normrnd(m2,s2,n,1);
    kk=find(u<pie_temp);
    ind_true=ones(n,1);
    ind_true(kk)=2;
    y(kk)=y2(kk);
    ncomp_max=n;
%Generate Data from DPP
else
    load test_data
    n=length(y);
    ncomp_max=n;
end
%Histogram of data
hist(y,25);
%Number of iterations for MCMC
nloop=10000;
warmup=1000;

%Parameters of inverse gammsa base distribution
G0_a=2;
G0_b=1;
%Parameters of normal base distribution
G0_m=0;
G0_c=1;
%Concentration parameter
alpha=1;
%Allocating space to number of components
ncomp=ones(nloop,1);

%Allocating space to phi's
mu=cell(ncomp_max);
sigmasq=cell(ncomp_max);
for j=1:ncomp_max
    mu{j}=ones(j,nloop+1);
    sigmasq{j}=ones(j,nloop+1);
end
%Random starting values for zz (indicators), assuming each compenent equally likely
ncomp(1)=2;
zz=ones(n,1);
u=rand(n,1);
sum1=zeros(n,1);
for j=1:ncomp(1)
       kk=find((u>sum1) & (u<sum1+ones(n,1)/ncomp(1)));
       zz(kk)=j;
       sum1=sum1+ones(n,1)/ncomp(1);
end
%MCMC Scheme
for p=1:nloop
    %Drawing the phi's
    for j=1:ncomp(p)
           j_ind=find(zz==j);
           n_j=length(j_ind);
           mu{ncomp(p)}(j,p+1)=normrnd(mean(y(j_ind)),sqrt(sigmasq{ncomp(p)}(j,p)/n_j*G0_c/(G0_c+1)));
           sigmasq{ncomp(p)}(j,p+1)=1./gamrnd(n_j/2+G0_a,1/(sum((y(j_ind)-mu{ncomp(p)}(j,p+1)).^2)/2+G0_b));
    end
    %Drawing the cluster indicators
    likelihood=zeros(ncomp(p)+1,1);
    prior_zz=zeros(ncomp(p)+1,1);
    for i=1:n
        z_not_i=zz;
        z_not_i(i)=[];
        for j=1:ncomp(p)
                n_not_i_j=sum(z_not_i==j);
                prior_zz(j)=(n_not_i_j)/(n-1+alpha);
                likelihood(j)=normpdf(y(i),mu{ncomp(p)}(j,p+1),sqrt(sigmasq{ncomp(p)}(j,p+1)));
        end
        prior_zz(j+1)=alpha/(n-1+alpha);
        bstar=y(i)^2/(2*(G0_c+1))+G0_b;
        likelihood(j+1)=G0_b^G0_a*gamma(G0_a+0.5)/gamma(G0_a)*bstar^(G0_a+0.5)/(2*pi*(G0_c+1));
        prob_zz=(prior_zz.*likelihood)./sum(prior_zz.*likelihood);
        zz(i)=find(mnrnd(1,prob_zz)==1);
    end
   %Updating number of components
    cc=unique(zz);
    ncomp(p+1)=length(cc); 
     %relabelling nonsequential categories
    for j=1:ncomp(p+1)
        kk=j;
        while cc(j)~=kk
            kk=kk+1;
        end
        ind=find(zz==kk);
        zz(ind)=j;
    end
            
end
figure
h=histogram(ncomp,'Binwidth',1);
%Histogram of phi, for the most likely
kk=find(ncomp==2);
plot(mu{2}(1,kk+1));
hold
plot(mu{2}(2,kk+1),'r');
