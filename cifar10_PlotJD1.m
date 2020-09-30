% 绘制数据集的 JD 曲线
% JD -- EPOCH
% cifar10-lenet={conv(1 4 7 10 12) relu(3 5 8 11) pool(2 6 9)};

% 绘制数据集的 JD 曲线
% JD -- EPOCH ----LayerIndex
% 
dataDir = 'cifar10-lenet-exp-jd1-6';
epoch = 0:1:30;
layerIndex = 1:1:12;

% JD-layerIndex 曲线
jdtrain = []; 
jdtest = [];
for layerId = layerIndex
    for ep = epoch
        dataName = sprintf('layer-%d-epoch-%d-JD.mat',layerId,ep);
        JD = load(fullfile(dataDir,dataName));
        JD = JD.JD;
        jdtrain(end+1) = JD.train;
        jdtest(end+1) = JD.test;
    end
    if true
        figure(layerId+100)
        plot(epoch,jdtrain,'mo-',epoch,jdtest,'b*-');
        xlabel('迭代次数-epoch');
        ylabel('类间KL散度-JD');
        legend('jd.train','jd.test');
        title(sprintf('类间KL散度~训练次数--layer(%d)',layerId));
    end
    jdtrain = [];
    jdtest = [];
end

% JD-EPOCH 曲线
epoch = [0,30];
jdtrain = []; 
jdtest = [];
for ep = epoch
    for layerId = layerIndex
        dataName = sprintf('layer-%d-epoch-%d-JD.mat',layerId,ep);
        JD = load(fullfile(dataDir,dataName));
        JD = JD.JD;
        jdtrain(end+1) = JD.train;
        jdtest(end+1) = JD.test;
    end
    if true
        figure(ep+1000)
        plot(layerIndex,jdtrain,'mo-',layerIndex,jdtest,'b*-');
        xlabel('网络层数-layerIndex');
        ylabel('类间KL散度-JD');
        legend('jd.train','jd.test');
        title(sprintf('类间KL散度~网络层数--epoch(%d)',ep));
    end
    jdtrain = [];
    jdtest = [];
end
