%準備資料
%請執行PrepareData_MG.m 或 PrepareData_PM25.m
%建議準備資料的程式碼要獨立成一個檔案，以確保每次亂數產生的結果會被保留，使BPNN與RBFNN使用到相同的資料，後續比較才不會受輸入訓練資料差異影響

%設定網路為BPNN架構
%net=newrb(data_in_Norm(:,train_index),data_real_out_Norm(:,train_index), 0.0,1.0,5,1);
net=newrbe(data_in_Norm(:,train_index),data_real_out_Norm(:,train_index), 1.0);

%網路模擬輸出值
data_out_Norm=sim(net,data_in_Norm);

%反正規化
data_out = data_out_Norm .* (data_real_out_max - data_real_out_min) + data_real_out_min;

%訓練資料繪圖
[train_rmse, train_R] = PlotResult(...
    data_out(:,train_index),...
    data_real_out(:,train_index),'訓練');
%驗證資料繪圖
[val_rmse, val_R] = PlotResult(...
    data_out(:,val_index),...
    data_real_out(:,val_index),'驗證');
%測試資料繪圖
[test_rmse, test_R] = PlotResult(...
    data_out(:,test_index),...
    data_real_out(:,test_index),'測試');

%ANN網路參數寫入檔案
save('RBFnet.mat',...
     'net','usingLabel',...
     'data_in_max','data_in_min',...
     'data_real_out_max','data_real_out_min');


