rng(1);

%generative process
alpha=5;
% K=10; 
K=2;
% N=50;
a=1; b=1;
% pi=drchrnd(alpha*ones(1,K),1)';

N=100;
nloop=10000;
nwarmup=5000;
nsim=100;

pi=[0.4 0.6]';
% lambda=gamrnd(a,1/b,K,1);
lambda=[2 6]';
    
for i=1:50    
    z=mnrnd(1,pi,N);
    y=gamrnd(1,1./(z*lambda));
    if i==1
        test_data=y;
    else
        test_data=[test_data y];
    end
end


for jj=1:50
    y=test_data(:,jj);
    a1=n+.5;
    b1=1/(sum(y)+.5);
    n_comp=ones(nloop+1-nwarmup,1);
    lambda=1/mean(y);
    for p=1:nloop
    %Model Selection step
        if(n_comp(p)==1) 
            n_comp_prop=2;
            n_comp_curr=n_comp(p);
            u1=rand;
            u2=rand;
            lam1_prop=(u2/u1)*lambda;
            lam0_prop=(1-u2)/(1-u1)*lambda;
            lam_curr=lambda;
            pie_prop=u2;
            f1_prop=lam1_prop*exp(-lam1_prop*y);
            f0_prop=lam0_prop*exp(-lam0_prop*y);
            log_target_prop=sum(log(pie_prop.*f1_prop+(1-pie_prop).*f0_prop))+...
            log(gampdf(lam1_prop,2.5,1))+log(gampdf(lam0_prop,2.5,1));
            log_target_curr=n*log(lam_curr)-lam_curr*sum(y)+log(gampdf(lambda,2.5,1));
            jac=abs(lam1_prop^2*lam0_prop^2/(lam_curr^3*pie_prop*(1-pie_prop)));
            met_rat(p)=exp(log_target_prop-log_target_curr)*jac;
            u=rand;
            if(u>met_rat(p))
                n_comp(p+1)=n_comp_curr;
            else
                n_comp(p+1)=n_comp_prop;
                theta_2=[lam1_prop lam0_prop pie_prop];
            end
        else
            n_comp_prop=1;
            n_comp_curr=n_comp(p);
            lam1_curr=theta_2(1);
            lam0_curr=theta_2(2);
            pie_curr=theta_2(3);
            lam_prop=1/(pie_curr/lam1_curr+(1-pie_curr)/lam0_curr);
            f1_curr=lam1_curr*exp(-lam1_curr*y);
            f0_curr=lam0_curr*exp(-lam0_curr*y);
            log_target_curr=sum(log(pie_curr.*f1_curr+(1-pie_curr).*f0_curr))+...
            log(gampdf(lam1_curr,2.5,1))+log(gampdf(lam0_curr,2.5,1));
            log_target_prop=n*log(lam_prop)-lam_prop*sum(y)+log(gampdf(lam_prop,2.5,1));
            jac=1/abs(lam1_curr^2*lam0_curr^2/(lam_prop^3*pie_curr*(1-pie_curr)));
            met_rat(p)=exp(log_target_prop-log_target_curr)*jac;
        u=rand;
        if(u<met_rat(p))
            n_comp(p+1)=n_comp_prop;
            lambda=lam_prop;
        else
            n_comp(p+1)=n_comp_curr;
        end
    end
end
kk=find(n_comp(nwarmup+1:nloop)==2);
pmod2_green(jj)=length(kk)/(nloop-nwarmup);
end