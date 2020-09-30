% mnist-lenet
% layer|   0|    1|    2|    3|    4|    5|    6|    7|    8|    9|     10|
% type|input| conv| relu|mpool| conv| relu|mpool| conv| relu| conv|softmxl|
% name|  n/a|conv1|relu1|pool1|conv2|relu2|pool2|conv3|relu3|conv4|    sml|

epoch = 0:1:30;
layerIndex = [2,3,5,6];
sets = {'Train','Test'};

calcFeatu = 1;
calcJD = 1;

% pp = gcp();
for layerId = layerIndex
    for ep = epoch
        if calcFeatu && (layerId~=5 && layerId~=6) || ep>20 
        for set = sets
            time = tic;
            %disp(sprintf('Satrt Train/Test! epcoh-%d:layerIndex-%d:set-%s',ep,layerId,cell2mat(set)));
            fprintf('Satrt Evaluation! epcoh-%d:layerIndex-%d:--->set-%s\n',ep,layerId,cell2mat(set));
            cd E:\MatConvNet-1.0-beta17\examples\mnist;
            cnn_mnist_test_jd1(cell2mat(set),ep,layerId);
            toc(time);
            fprintf('End! -----%s Done!!-----\n',cell2mat(set)); 
        end
        end
    end
end
% delete(pp);

for layerId = layerIndex
    for ep = epoch        
        if calcJD && (layerId~=5 && layerId~=6) || ep>20
            time = tic;
            fprintf('Satrt JD! epcoh-%d:layerIndex-%d\n',ep,layerId);
            cd E:\CNN1_FeatureEvluation;
            mnist_CalcJD1O(ep,layerId);
            fprintf('End! -----JD Done!!-----\n\n'); 
            toc(time);
        end
    end
end

% delete(pp);
