% �������ݼ��� JD ����
% JD -- EPOCH --- LayerIndex
% 
% JD-layerIndex ����
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
        xlabel('��������-epoch');
        ylabel('���KLɢ��-JD');
        legend('jd.val');
        title(sprintf('���KLɢ��~ѵ������-@-layer(%d)',layerId));
    end
    jdtrain = [];
    jdval = [];
end

% JD-EPOCH ����
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
        xlabel('�������-layerIndex');
        ylabel('���KLɢ��-JD');
        legend('jd.val');
        title(sprintf('���KLɢ��~�������-@-epoch(%d)',ep));
    end
    jdval = [];
end

% 
