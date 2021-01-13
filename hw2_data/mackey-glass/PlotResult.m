function [rmse, R] = PlotResult( data_out, data_real_out,StateString)
%PlotResult �e���G��
%   �]�t�ǦC�ϡB�j�k���R��

%RMSE�p��
rmse=sqrt(mse(data_out-data_real_out));
%R�p��
R=corr(data_out',data_real_out');

%��ƥؼп�X�ȻP������X�Ȫ��ǦC��
figure
plot(data_real_out,'b-');
hold on
plot(data_out,'rx');
title(strcat(StateString,...
     '�ǦC�� RMSE=',num2str(rmse)))
legend('�ؼп�X��','������X��');
xlabel('��ƧǦC');ylabel('f(x,y)');

%�u�ʰj�k���R��
figure
plotregression(data_real_out,data_out,strcat(StateString,' Regression'));

end

