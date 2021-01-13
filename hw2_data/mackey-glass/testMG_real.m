clc;
%準備資料
test_data_real_out = LoadData_MG_Y('test_Y_MG_real.txt');
test_data_real_out = table2array(test_data_real_out(:,{'T6'}))';
%測試資料繪圖
[test_rmse, test_R] = PlotResult(...
    test_data_out,...
    test_data_real_out,'測試');