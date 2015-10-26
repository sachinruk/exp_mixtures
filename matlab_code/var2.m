function val=var2(y,mean)

val=sum((y-mean).^2)/(length(y)-1);