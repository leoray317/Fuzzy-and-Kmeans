function [rmse, R] = PlotResult( data_out, data_real_out,StateString)
%PlotResult 畫結果圖
%   包含序列圖、迴歸分析圖

%RMSE計算
rmse=sqrt(mse(data_out-data_real_out));
%R計算
R=corr(data_out',data_real_out');

%資料目標輸出值與網路輸出值的序列圖
figure
plot(data_real_out,'b-');
hold on
plot(data_out,'rx');
title(strcat(StateString,...
     '序列圖 RMSE=',num2str(rmse)))
legend('目標輸出值','網路輸出值');
xlabel('資料序列');ylabel('f(x,y)');

%線性迴歸分析圖
figure
plotregression(data_real_out,data_out,strcat(StateString,' Regression'));

end

