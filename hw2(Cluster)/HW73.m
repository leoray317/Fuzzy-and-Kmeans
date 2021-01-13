clc
clear
a1=(-pi/4)+(pi/2)*rand(1,20);
b1=2*pi*rand(1,20);
sdev1=0.6*rand(1,20);
sdev2=0.6*rand(1,20);
output1=x1(a1,b1)+sdev1;
output2=y1(a1,b1)+sdev2;

a2=(pi/4)+((2*pi)/4)*rand(1,20);
b2=(-pi/2)+(pi)*rand(1,20);
sdev3=0.6*rand(1,20);
sdev4=0.6*rand(1,20);
output3=x2(a2,b2)+sdev3;
output4=y2(a2,b2)+sdev4;

a3=(-pi/2)+((4*pi)/2)*rand(1,20);
b3=(-pi/4)+((4*pi)/4)*rand(1,20);
sdev5=0.6*rand(1,20);
sdev6=0.6*rand(1,20);
output5=x3(a3,b3)+sdev5;
output6=y3(a3,b3)+sdev6;

