% DPP
% ============================================================
% Generate Data
n=100;
l1=1;%mean of exp distribution 1
l2=3;%mean of exp distribution 2
pie_temp=0.5;
u=rand(n,1);
y=exprnd(1/l1,n,1);%Note the parameterization of the exponetial distribution in matlab
y2=exprnd(1/l2,n,1);
kk=find(u<pie_temp);
ind_true=ones(n,1);
ind_true(kk)=2;
y(kk)=y2(kk);
n=length(y);
ncomp_max=n;
%Histogram of data
hist(y,25);
%Number of iterations for MCMC
nloop=10000;
warmup=1000;

%Parameters of base distribution
G0_a=1;
G0_b=1;
%Concentration parameter
alpha=1;
%Allocating space to number of components
ncomp=ones(nloop,1);

%The true log posterior density
x1=linspace(0.01,5);
x2=linspace(0.01,5);
post_dens_true=zeros(100,100);
for j=1:100
    for k=1:100
        post_dens_true(j,k)=sum(log(exppdf(y,1./x1(j))*pie_temp+(1-pie_temp)*exppdf(y,1./x2(k))))...
            +log(gampdf(x1(j),G0_a,1./G0_b))+log(gampdf(x2(k),G0_a,1./G0_b));
        %Note that matlab paramertizes a gamma distribution with b=1/b.
    end
end
figure
surf(x2,x1,exp(post_dens_true-max(max(post_dens_true))))
%Allocating space to phi's
phi=cell(ncomp_max);
for j=1:ncomp_max
    phi{j}=ones(j,nloop+1);
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
           phi{ncomp(p)}(j,p+1)=gamrnd(n_j+G0_a,1/(sum(y(j_ind))+G0_b));
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
                likelihood(j)=phi{ncomp(p)}(j,p+1)*exp(-y(i)*phi{ncomp(p)}(j,p+1));
        end
        prior_zz(j+1)=alpha/(n-1+alpha);
        likelihood(j+1)=G0_b^G0_a*G0_a/((y(i)+G0_b)^(G0_a+1));
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
plot(phi{2}(1,kk+1));
hold
plot(phi{2}(2,kk+1),'r');
