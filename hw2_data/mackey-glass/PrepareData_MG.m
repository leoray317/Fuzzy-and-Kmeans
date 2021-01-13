clc;
clear;
%準備資料
data_in = LoadData_MG_X('MackeyGlass\train_X.txt');
data_real_out = LoadData_MG_Y('MackeyGlass\train_Y.txt');
usingLabel={'T18','T12','T6','T0'};
data_in = table2array(data_in(:,usingLabel))';
data_real_out = table2array(data_real_out(:,{'T6'}))';

%訓練，驗證，測試資料索引值設定
A=1:166;
B=166:333;
C=334:500;
%1
train_index =  [A,B];
val_index   = C;
test_index  = C;
%2
% train_index =  [A,C] ;
% val_index   = B;
% test_index  = B;
%3
% train_index =   [B,C];
% val_index   = A;
% test_index  = A;


%資料正規化
data_in_max =  max(data_in(:,train_index),[],2);
data_in_min =  min(data_in(:,train_index),[],2);
data_in_Norm = (data_in - data_in_min) ./ (data_in_max - data_in_min);

data_real_out_max =  max(data_real_out(:,train_index),[],2);
data_real_out_min =  min(data_real_out(:,train_index),[],2);
data_real_out_Norm = (data_real_out - data_real_out_min) ./ (data_real_out_max - data_real_out_min);