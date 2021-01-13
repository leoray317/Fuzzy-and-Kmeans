%�ǳƸ��
%�а���PrepareData_MG.m �� PrepareData_PM25.m
%��ĳ�ǳƸ�ƪ��{���X�n�W�ߦ��@���ɮסA�H�T�O�C���üƲ��ͪ����G�|�Q�O�d�A��BPNN�PRBFNN�ϥΨ�ۦP����ơA�������~���|����J�V�m��Ʈt���v�T

%�]�w������BPNN�[�c
net=feedforwardnet(10,'trainlm');

%�M�θ�ƽd��
net=configure(net,data_in_Norm,data_real_out_Norm);

%�]�w�v���Ѽƪ�l�Ƥ�k
net.inputweights{1,1}.initFcn = 'rands';
net.biases{1,1}.initFcn = 'rands';

%�]�w��Ƥ��դ覡
net.divideFcn = 'divideind';
net.divideParam.trainInd = train_index;
net.divideParam.valInd   = val_index;
net.divideParam.testInd  = test_index;

%�����l��
net=init(net);

%�]�w�V�m�Ѽ�
net.trainParam.epochs = 50;
net.trainParam.goal = 0.005;

%��������V�m
net=train(net,data_in_Norm,data_real_out_Norm);

%����������X��
data_out_Norm = sim(net,data_in_Norm);

%�ϥ��W��
data_out = data_out_Norm .* (data_real_out_max - data_real_out_min) + data_real_out_min;
 
%�V�m���ø��
[train_rmse, train_R] = PlotResult(...
    data_out(:,train_index),...
    data_real_out(:,train_index),'�V�m');
%���Ҹ��ø��
[val_rmse, val_R] = PlotResult(...
    data_out(:,val_index),...
    data_real_out(:,val_index),'����');
%���ո��ø��
[test_rmse, test_R] = PlotResult(...
    data_out(:,test_index),...
    data_real_out(:,test_index),'����');

%ANN�����ѼƼg�J�ɮ�
save('BPNNnet.mat',...
     'net','usingLabel',...
     'data_in_max','data_in_min',...
     'data_real_out_max','data_real_out_min');