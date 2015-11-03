function nmi_=nmi(l,c)
%%%%%%%%%%%%%%%%%%%%5
% Normalised Mutual Information
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Inputs
% l = (true) labels
% c = (assigned) clusters

%Create confusion matrix
c=relabel(c);
l=relabel(l);
confusion_matrix=zeros(max(l),max(c));

for i=1:max(l)
    for j=1:max(c)
        confusion_matrix(i,j)=sum(c(l==i)==j);
    end
end

N=length(l);
l_sum=sum(confusion_matrix,2);
c_sum=sum(confusion_matrix,1);

%mutual information
I=0;
for i=1:max(l)
    for j=1:max(c)
        if confusion_matrix(i,j)==0
            continue;
        end        
        I=I+(confusion_matrix(i,j)/N)*log(N*confusion_matrix(i,j)/(l_sum(i)*c_sum(j)));
    end
end

%entropy
H_l=-sum((l_sum/N).*log(l_sum/N));
H_c=-sum((c_sum/N).*log(c_sum/N));

nmi_=2*I/(H_l+H_c);