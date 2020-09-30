% �������ݼ��� JD ����
% JD -- EPOCH
% cifar10-lenet={conv(1 4 7 10 12) relu(3 5 8 11) pool(2 6 9)};

% �������ݼ��� JD ����
% JD -- EPOCH ----LayerIndex
% 
dataDir = 'cifar10-lenet-exp-jd1-6';
epoch = 0:1:30;
layerIndex = 1:1:12;

% JD-layerIndex ����
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
        xlabel('��������-epoch');
        ylabel('���KLɢ��-JD');
        legend('jd.train','jd.test');
        title(sprintf('���KLɢ��~ѵ������--layer(%d)',layerId));
    end
    jdtrain = [];
    jdtest = [];
end

% JD-EPOCH ����
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
        xlabel('�������-layerIndex');
        ylabel('���KLɢ��-JD');
        legend('jd.train','jd.test');
        title(sprintf('���KLɢ��~�������--epoch(%d)',ep));
    end
    jdtrain = [];
    jdtest = [];
end
