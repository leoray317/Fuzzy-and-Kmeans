clc;
%�ǳƸ��
test_data_real_out = LoadData_PM25_Y('test_Y_PM25_real.txt');
test_data_real_out = table2array(test_data_real_out(:,{'t0PM25'}))';
%���ո��ø��
[test_rmse, test_R] = PlotResult(...
    test_data_out,...
    test_data_real_out,'����');