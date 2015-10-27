function y=fillLast(y)

fillVal=nan;
for i=1:length(y)
    if y(i) %non-zero value
        fillVal=y(i);
    else
        y(i)=fillVal;
    end
end