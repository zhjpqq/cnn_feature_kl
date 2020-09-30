
% cifar-lenet
% layer|   0|    1|      2|   3|    4|   5|      6|    7|   8|      9|   10|  11|  12|     13|
% type|input| conv|  mpool|relu| conv|relu|  apool| conv|relu|  apool| conv|relu|conv|softmxl|

epoch = 0:1:30;  % [1:1:17,18:2:40]
layerIndex = 12:-1:1;
sets = {'Train','Test'};
calcFeatu = 0;
calcJD = 1;

for layerId = layerIndex 
    for ep = epoch
        if calcFeatu
            for set = sets
                time = tic;
                fprintf('Satrt Evaluation! epcoh-%d:layerIndex-%d:--->set-%s\n',ep,layerId,cell2mat(set));
                cd E:\MatConvNet-1.0-beta17\examples\cifar;
                cnn_cifar_test_jd1(cell2mat(set),ep,layerId);
                toc(time);
                fprintf('End! -----%s Done!!-----\n',cell2mat(set)); 
            end
        end
        if calcJD
            time = tic;
            fprintf('Satrt JD! epcoh-%d:layerIndex-%d',ep,layerId);
            cd E:\CNN1_FeatureEvluation;
            cifar10_CalcJD1(ep,layerId);
            toc(time);
            fprintf('End! -----JD Done!!-----\n\n'); 
        end
    end
end



% cd E:\MatConvNet-1.0-beta17\examples\cifar;
% for i = 0:2:60
%     for j = {'Train','Test'}
%         cnn_cifarX_test(j,i);
%     end
% end
% 
% cd E:\CNN1_FeatureEvluation;
% for epoch = 0:2:60
%     cifar10_Main(epoch);
% end
% 
