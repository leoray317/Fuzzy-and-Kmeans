clc;
%準備資料
test_data_real_out = LoadData_PM25_Y('test_Y_PM25_real.txt');
test_data_real_out = table2array(test_data_real_out(:,{'t0PM25'}))';
%測試資料繪圖
[test_rmse, test_R] = PlotResult(...
    test_data_out,...
    test_data_real_out,'測試');