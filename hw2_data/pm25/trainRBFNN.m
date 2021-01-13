%�ǳƸ��
%�а���PrepareData_MG.m �� PrepareData_PM25.m
%��ĳ�ǳƸ�ƪ��{���X�n�W�ߦ��@���ɮסA�H�T�O�C���üƲ��ͪ����G�|�Q�O�d�A��BPNN�PRBFNN�ϥΨ�ۦP����ơA�������~���|����J�V�m��Ʈt���v�T

%�]�w������BPNN�[�c
%net=newrb(data_in_Norm(:,train_index),data_real_out_Norm(:,train_index), 0.0,1.0,5,1);
net=newrbe(data_in_Norm(:,train_index),data_real_out_Norm(:,train_index), 1.0);

%����������X��
data_out_Norm=sim(net,data_in_Norm);

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
save('RBFnet.mat',...
     'net','usingLabel',...
     'data_in_max','data_in_min',...
     'data_real_out_max','data_real_out_min');


