RADII=0.45;
[C,S]=subclust(X,RADII);
figure;
plot(X(:,1),X(:,2),'m.');
hold on
plot(C(:,1),C(:,2),'+','markersize',25);
xlabel('X');
ylabel('Y');
title('subtractive');