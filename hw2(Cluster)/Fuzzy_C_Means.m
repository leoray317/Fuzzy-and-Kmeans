clc
centerNum=4;
[center,U,objFcn]=fcm(X,centerNum);
maxU=max(U);
mark={'o','x','s','*','^','d','v','.','<','>'};
figure
for i=1:centerNum;
    index{i}=find(U(i,:)==maxU);
    plot(X(index{i},1),X(index{i},2)....
        ,strcat('m',char(mark(i))),'markersize',5);
    hold on
end
plot(center(:,1),center(:,2),'k+','markersize',10);
xlabel('X');
ylabel('Y');
title('Fuzzy_C_Means');