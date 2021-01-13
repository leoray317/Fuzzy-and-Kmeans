clc
X=[output1 output3 output5;output2 output4 output6]';
plot(X(:,1),X(:,2),'.')
[Idx,centers] = kmeans(X,4)
idx1=find(Idx==1);
idx2=find(Idx==2);
idx3=find(Idx==3);
idx4=find(Idx==4);
plot(X(idx1,1),X(idx1,2),'b.');
hold on
plot(X(idx2,1),X(idx2,2),'ro');
plot(X(idx3,1),X(idx3,2),'gd');
plot(X(idx4,1),X(idx4,2),'y*');
plot(centers(:,1),centers(:,2),'k+');
title('kmeans');