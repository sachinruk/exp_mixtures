function idx=choose_idx(num,len)

if len==1
    idx=true;
else
    idx1=randsample(len,num);
    idx=false(1,len);
    idx(idx1)=true;
end