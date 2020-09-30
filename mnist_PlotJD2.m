% 绘制数据集的 JD 曲线
% JD -- EPOCH
% 单类样本的收敛性曲线绘制

% JD-Batch 曲线
dataDir = 'mnist-lenet-exp-jd3';
epoch = 1;
layerIndex = 9:-1:1;
batchIndex = 0:20:600;
sets = {'Train','Test'};
jdtrain = []; 
jdtest = [];
xep = [];
for la = layerIndex
    for ep = epoch
        for ba = batchIndex
            if ep>min(epoch) && ba ==0
                % 前端跳过
                continue;
            end
            if ep==max(epoch) && ba==600
                % 末端跳过
                break;
            end
            dataName = sprintf('layer-%d-epoch-%d-batch-%d-JD.mat',la,ep,ba);
            JD = load(fullfile(dataDir,dataName));
            JD = JD.JD;
            jdtrain(end+1) = JD.train;
            jdtest(end+1) = JD.test;
            xep(end+1) = (ep-1)*600 + ba;
        end
    end
    if true
        figure(la+100)
        plot(xep,jdtrain,'mo-',xep,jdtest,'b*-');
        xlabel('迭代批次-batch');
        ylabel('类内KL散度-JD');
        legend('jd.train','jd.test');
        title(sprintf('类内KL散度收敛曲线~训练次数-@-layer(%d)',la));
    end
    xep=[];
    jdtrain = [];
    jdtest = [];
end
