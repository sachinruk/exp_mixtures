function idx=choose_idx(num,len)

idx1=randsample(len,num);
idx=false(1,len);
idx(idx1)=true;