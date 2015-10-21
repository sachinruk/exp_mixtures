function z_new=lof(z_current)

% [N,K]=size(z_current);
history=bi2de(z_current','left-msb');
[~, idx]= sort(history,'descend');
z_new = z_current(:,idx);