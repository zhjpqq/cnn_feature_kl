% mnist-lenet
% layer|   0|    1|    2|    3|    4|    5|    6|    7|    8|    9|     10|
% type|input| conv| relu|mpool| conv| relu|mpool| conv| relu| conv|softmxl|
% name|  n/a|conv1|relu1|pool1|conv2|relu2|pool2|conv3|relu3|conv4|    sml|

% 计算逐个batch下的特征向量
% 用以评估类内散度的收敛性

calcFeature = false;
calcJd = true;

%计算特征图
if calcFeature
epoch = 1:1:2;
batchIndex = 0:20:600;
layerIndex = 4:-1:0;
sets = {'Train','Test'};
for layerId = layerIndex
    for ep = epoch
        for batchId = batchIndex
            if ep>1 && batchId ==0
                continue;
            end
            for set = sets
                time = tic;
                %disp(sprintf('Satrt Train/Test! epcoh-%d:layerIndex-%d:set-%s',ep,layerId,cell2mat(set)));
                fprintf('Satrt Train/Test! epcoh-%d:layerIndex-%d:batchIndex-%d:set-%s\n',ep,layerId,batchId,cell2mat(set));
                cd E:\MatConvNet-1.0-beta17\examples\mnist;
                cnn_mnist_test_jd2(cell2mat(set),ep,layerId,batchId);
                fprintf('\nEnd! -----Train/Test Done!!-----\n'); 
                toc(time);
            end
        end
    end
end
end

%计算散度
if calcJd
epoch = 1:1:2;
batchIndex = 0:20:600;
layerIndex = 3:-1:0;
sets = {'Train','Test'};
overep = 0;
for layerId = layerIndex
    for ep = epoch
        for batchId = batchIndex
            if ep>1 && batchId == 0
                continue;
            end
            time = tic;
            fprintf('Satrt JD! epcoh-%d:layerIndex-%d:batchId-%d\n',ep,layerId,batchId);
            cd E:\CNN1_FeatureEvluation;
            if batchId ==0 || ~(mod(batchId,600)==0)
                %第0个batch和第599个batch都在本epoch内循环
                epn = ep;
                batchIdn = batchId + 20;
            else
                %到达当前epoch的第600个batch，进行跨epcoh循环
                epn = ep + 1;
                if epn > max(epoch)
                    overep = 1;
                    break;
                else 
                    overep = 0;
                end
                batchIdn = 20;
            end
            mnist_CalcJD2(layerId,ep,epn,batchId,batchIdn);
            fprintf('\nEnd! -----JD Done!!-----\n'); 
            toc(time);
        end
        if overep
            %跳到下一个 layerIndex
            break;
        end        
    end
end
end
