function class=relabel(class)

u_classes=unique(class,'stable');
max_class=max(u_classes);
for i=1:length(u_classes)
    class(class==u_classes(i))=max_class+i;
end
class=class-max_class;