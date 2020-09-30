% �������ݼ��� JD ����
% JD -- EPOCH
% ���epoch�£����ɢ������ 
% JD-layerIndex
% ���layer, ���ɢ������

run(fullfile('E:\MatConvNet-1.0-beta17\matlab', 'vl_setupnn.m')) ;

% JD-layerIndex ����

% JD-EPOCH ����
dataDir = 'cifar10-lenet-exp-jd1-6';
epoch = [0,1,30];
layerIndex = 1:1:12;

% �����������ͼά��
net = cnn_cifar_init_jdperdim();
netInfo = vl_simplenn_display(net);
featuSize = netInfo.dataSize;
featuDim = zeros(1,numel(layerIndex)-1);
for lidx = 2 : numel(layerIndex)    
    featuDim(1,lidx) = prod(featuSize(:,lidx));
end

% ��ȡ��������ͼɢ��ֵ
% ���Ƹ���ɢ��~layerIndex@epoch
% �������ɢ��ѹ����=�ò�ɢ��ֵ/�ò�����ά��

jdtrain = []; 
jdtest = [];
jdtrainPerDim = [];
jdtestPerDim = [];
for ep = epoch
    for layerId = layerIndex
        dataName = sprintf('layer-%d-epoch-%d-JD.mat',layerId,ep);
        JD = load(fullfile(dataDir,dataName));
        JD = JD.JD;
        jdtrain(end+1) = JD.train;
        jdtest(end+1) = JD.test;
    end
    if false
        figure(ep+1000)
        plot(layerIndex,jdtrain,'mo-',layerIndex,jdtest,'b*-');
        xlabel('�������-layerIndex');
        ylabel('���KLɢ��-JD');
        legend('jd.train','jd.test');
        title(sprintf('���KLɢ��~�������-@-epoch(%d)',ep));
    end
    if true
        jdtrainPerDim = jdtrain./featuDim;
        jdtestPerDim = jdtest./featuDim;
        figure(ep+10000)
        plot(layerIndex,jdtrainPerDim,'mo-',layerIndex,jdtestPerDim,'b*-');
        xlabel('�������-layerIndex');
        ylabel('���KLɢ��/����ά��-JD');
        legend('jd.train','jd.test');
        title(sprintf('���������KLɢ��~�������-@-epoch(%d)',ep));
    end
    jdtrainPerDim = [];
    jdtestPerDim = [];
    jdtrain= [];
    jdtest = [];
end

% ���epoch�ĵ�����KLɢ�Ȼ���һ��ͼ��
eps = numel(epoch);
layers = numel(layerIndex);
jdtrain = zeros(eps,layers);
jdtest = zeros(eps,layers);
epid = 1;
for ep = epoch
    for layerId = layerIndex
        dataName = sprintf('layer-%d-epoch-%d-JD.mat',layerId,ep);
        JD = load(fullfile(dataDir,dataName));
        JD = JD.JD;
        jdtrain(epid,layerId) = JD.train;
        jdtest(epid,layerId) = JD.test;    
    end 
    epid = epid + 1;
end
jdtrainPerDim = jdtrain./repmat(featuDim,eps,1);
jdtestPerDim = jdtest./repmat(featuDim,eps,1);
figure(1010101)
plot(layerIndex,jdtrainPerDim,layerIndex,jdtestPerDim);
xlabel('�������-layerIndex');
ylabel('���KLɢ��/����ά��-JD');
legend('jd.train','jd.test');
title(sprintf('���������KLɢ��~�������-@-epoch(%d)',ep));

% xlswrite('cifar_jdperdim_train',jdtrainPerDim);
% xlswrite('cifar_jdperdim_test',jdtestPerDim);
