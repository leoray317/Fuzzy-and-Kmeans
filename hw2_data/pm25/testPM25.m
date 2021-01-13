%�ǳƸ��
test_data = LoadData_PM25_X('PM25\test_X.txt');
test_data_ID = table2array(test_data(:,'ID'))';
test_data_in = table2array(test_data(:,usingLabel))';
test_data_in_Norm = (test_data_in - data_in_min) ./ (data_in_max - data_in_min);

%����������X��
test_data_out_Norm = sim(net,test_data_in_Norm);

%�ϥ��W��
test_data_out = test_data_out_Norm .* (data_real_out_max - data_real_out_min) + data_real_out_min;

%�ƾڼg�J�ɮ�
fileID =fopen('test_Y.txt','w');
fprintf(fileID,'ID\tt0-PM2.5\n');
for i=1:length(test_data_ID)
    fprintf(fileID,'%0.0f\t%0.2f\n',...
            test_data_ID(i),test_data_out(i));
end
fclose(fileID);