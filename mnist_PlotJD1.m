% �������ݼ��� JD ����
% JD -- EPOCH
% ���epoch�£����ɢ������ 
% JD-layerIndex
% ���layer, ���ɢ������

% JD-layerIndex ����
dataDir = 'mnist-lenet-exp-jd1-4';
epoch = 1:20;
layerIndex = [1,2,3,4];
sets = {'Train','Test'};
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
        title(sprintf('���KLɢ��~ѵ������-@-layer(%d)',layerId));
    end
    jdtrain = [];
    jdtest = [];
end

% JD-EPOCH ����

% epoch = [0,20];
% % layerIndex = 4:1:9;
% % sets = {'Train','Test'};
% jdtrain = []; 
% jdtest = [];
% for ep = epoch
%     for layerId = layerIndex
%         dataName = sprintf('layer-%d-epoch-%d-JD.mat',layerId,ep);
%         JD = load(fullfile(dataDir,dataName));
%         JD = JD.JD;
%         jdtrain(end+1) = JD.train;
%         jdtest(end+1) = JD.test;
%     end
%     if true
%         figure(ep+1000)
%         plot(layerIndex,jdtrain,'mo-',layerIndex,jdtest,'b*-');
%         xlabel('�������-layerIndex');
%         ylabel('���KLɢ��-JD');
%         legend('jd.train','jd.test');
%         title(sprintf('���KLɢ��~�������-@-epoch(%d)',ep));
%     end
%     jdtrain = [];
%     jdtest = [];
% end
% 
% 
