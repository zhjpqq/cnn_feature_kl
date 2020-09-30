% 绘制数据集的 JD 曲线
% JD -- EPOCH --- LayerIndex
% 
% JD-layerIndex 曲线
dataDir = 'imagenet-alexnet-exp';
epoch = [1000 0];
layerIndex = 10:1:20;
set = {'val'};
jdtrain = []; 
jdval = [];
for layerId = layerIndex
    for ep = epoch
        dataName = sprintf('layer-%d-epoch-%d-%s-JD.mat',layerId,ep,cell2mat(set));
        JD = load(fullfile(dataDir,dataName));
        JD = JD.JD;
        jdval(end+1) = JD;
    end
    if true
        figure(layerId+100)
        plot(epoch,jdval,'b*-');
        xlabel('迭代次数-epoch');
        ylabel('类间KL散度-JD');
        legend('jd.val');
        title(sprintf('类间KL散度~训练次数-@-layer(%d)',layerId));
    end
    jdtrain = [];
    jdval = [];
end

% JD-EPOCH 曲线
dataDir = 'imagenet-alexnet-exp';
epoch = [1000 0];
layerIndex = 10:1:20;
sets = {'Val'};
jdval = [];
for ep = epoch
    for layerId = layerIndex
        dataName = sprintf('layer-%d-epoch-%d-%s-JD.mat',layerId,ep,cell2mat(set));
        JD = load(fullfile(dataDir,dataName));
        JD = JD.JD;
        jdval(end+1) = JD;
    end
    if true
        figure(ep+1000)
        plot(layerIndex,jdval,'b*-');
        xlabel('网络层数-layerIndex');
        ylabel('类间KL散度-JD');
        legend('jd.val');
        title(sprintf('类间KL散度~网络层数-@-epoch(%d)',ep));
    end
    jdval = [];
end

% 
