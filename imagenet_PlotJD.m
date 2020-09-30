% �������ݼ��� JD ����
% JD -- EPOCH --- LayerIndex
% 
% JD-layerIndex ����
dataDir = 'imagenet-alexnet-exp';
epoch = [1000 0];
layerIndex = 10:1:20;
jdtrain = []; 
jdval = [];
for layerId = layerIndex
    for ep = epoch
        dataName = sprintf('layer-%d-epoch-%d-JD.mat',layerId,ep);
        JD = load(fullfile(dataDir,dataName));
        JD = JD.JD;
        jdtrain(end+1) = JD.train;
        jdval(end+1) = JD.val;
    end
    if true
        figure(layerId+100)
        plot(epoch,jdtrain,'mo-',epoch,jdval,'b*-');
        xlabel('��������-epoch');
        ylabel('���KLɢ��-JD');
        legend('jd.train','jd.val');
        title(sprintf('���KLɢ��~ѵ������-@-layer(%d)',layerId));
    end
    jdtrain = [];
    jdval = [];
end

% JD-EPOCH ����
dataDir = 'imagenet-alexnet-exp';
epoch = [1000 0];
layerIndex = 16:1:20;
sets = {'Train','Val'};
jdtrain = []; 
jdval = [];
for ep = epoch
    for layerId = layerIndex
        dataName = sprintf('layer-%d-epoch-%d-JD.mat',layerId,ep);
        JD = load(fullfile(dataDir,dataName));
        JD = JD.JD;
        jdtrain(end+1) = JD.train;
        jdval(end+1) = JD.val;
    end
    if true
        figure(ep+1000)
        plot(layerIndex,jdtrain,'mo-',layerIndex,jdval,'b*-');
        xlabel('�������-layerIndex');
        ylabel('���KLɢ��-JD');
        legend('jd.train','jd.test');
        title(sprintf('���KLɢ��~�������-@-epoch(%d)',ep));
    end
    jdtrain = [];
    jdval = [];
end


