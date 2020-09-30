% �������ݼ��� JD ����
% JD -- EPOCH
% �������������������߻���

% JD-Batch ����
dataDir = 'cifar10-lenet-exp-jd2';
epoch = 1;
layerIndex = 12:-1:2;
batchIndex = 0:20:500;
sets = {'Train','Test'};
jdtrain = []; 
jdtest = [];
xep = [];
for la = layerIndex
    for ep = epoch
        for ba = batchIndex
            if ep>min(epoch) && ba ==0
                % ǰ������
                continue;
            end
            if ep==max(epoch) && ba==500
                % ĩ������
                break;
            end
            dataName = sprintf('layer-%d-epoch-%d-batch-%d-JD.mat',la,ep,ba);
            JD = load(fullfile(dataDir,dataName));
            JD = JD.JD;
            jdtrain(end+1) = JD.train;
            jdtest(end+1) = JD.test;
            xep(end+1) = (ep-1)*500 + ba;
        end
    end
    if true
        figure(la+100)
        plot(xep,jdtrain,'mo-',xep,jdtest,'b*-');
        xlabel('��������-batch');
        ylabel('����KLɢ��-JD');
        legend('jd.train','jd.test');
        title(sprintf('����KLɢ����������~ѵ������-@-layer(%d)',la));
    end
    xep = [];
    jdtrain = [];
    jdtest = [];
end
