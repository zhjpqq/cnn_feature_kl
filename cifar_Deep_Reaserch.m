% 绘制数据集的 JD 曲线
% JD -- EPOCH
% 逐个epoch下，类间散度曲线 
% JD-layerIndex
% 逐个layer, 类间散度曲线

run(fullfile('E:\MatConvNet-1.0-beta17\matlab', 'vl_setupnn.m')) ;

% JD-layerIndex 曲线

% JD-EPOCH 曲线
dataDir = 'cifar10-lenet-exp-jd1-6';
epoch = [0,1,30];
layerIndex = 1:1:12;

% 计算各层特征图维数
net = cnn_cifar_init_jdperdim();
netInfo = vl_simplenn_display(net);
featuSize = netInfo.dataSize;
featuDim = zeros(1,numel(layerIndex)-1);
for lidx = 2 : numel(layerIndex)    
    featuDim(1,lidx) = prod(featuSize(:,lidx));
end

% 获取各层特征图散度值
% 绘制各层散度~layerIndex@epoch
% 计算各层散度压缩率=该层散度值/该层特征维度

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
        xlabel('网络层数-layerIndex');
        ylabel('类间KL散度-JD');
        legend('jd.train','jd.test');
        title(sprintf('类间KL散度~网络层数-@-epoch(%d)',ep));
    end
    if true
        jdtrainPerDim = jdtrain./featuDim;
        jdtestPerDim = jdtest./featuDim;
        figure(ep+10000)
        plot(layerIndex,jdtrainPerDim,'mo-',layerIndex,jdtestPerDim,'b*-');
        xlabel('网络层数-layerIndex');
        ylabel('类间KL散度/特征维度-JD');
        legend('jd.train','jd.test');
        title(sprintf('单特征类间KL散度~网络层数-@-epoch(%d)',ep));
    end
    jdtrainPerDim = [];
    jdtestPerDim = [];
    jdtrain= [];
    jdtest = [];
end

% 多个epoch的单特征KL散度画入一张图内
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
xlabel('网络层数-layerIndex');
ylabel('类间KL散度/特征维度-JD');
legend('jd.train','jd.test');
title(sprintf('单特征类间KL散度~网络层数-@-epoch(%d)',ep));

% xlswrite('cifar_jdperdim_train',jdtrainPerDim);
% xlswrite('cifar_jdperdim_test',jdtestPerDim);
