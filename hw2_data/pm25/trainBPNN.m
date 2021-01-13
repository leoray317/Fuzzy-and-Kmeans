%準備資料
%請執行PrepareData_MG.m 或 PrepareData_PM25.m
%建議準備資料的程式碼要獨立成一個檔案，以確保每次亂數產生的結果會被保留，使BPNN與RBFNN使用到相同的資料，後續比較才不會受輸入訓練資料差異影響

%設定網路為BPNN架構
net=feedforwardnet(10,'trainlm');

%套用資料範圍
net=configure(net,data_in_Norm,data_real_out_Norm);

%設定權重參數初始化方法
net.inputweights{1,1}.initFcn = 'rands';
net.biases{1,1}.initFcn = 'rands';

%設定資料分組方式
net.divideFcn = 'divideind';
net.divideParam.trainInd = train_index;
net.divideParam.valInd   = val_index;
net.divideParam.testInd  = test_index;

%執行初始化
net=init(net);

%設定訓練參數
net.trainParam.epochs = 50;
net.trainParam.goal = 0.005;

%執行網路訓練
net=train(net,data_in_Norm,data_real_out_Norm);

%網路模擬輸出值
data_out_Norm = sim(net,data_in_Norm);

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
save('BPNNnet.mat',...
     'net','usingLabel',...
     'data_in_max','data_in_min',...
     'data_real_out_max','data_real_out_min');