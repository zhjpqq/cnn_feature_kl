% imagenet- alexnet

epoch = [1000];   %[1000,0]; 
layerIndex =  9:-1:5; %20:-1:16;
sets = {'Val'};
calcJD = 1;
calcFeatureMap = 1;

for ep = epoch
    for layerId = layerIndex
        if calcFeatureMap
            for set = sets
                time = tic;
                %disp(sprintf('Satrt Train/Test! epcoh-%d:layerIndex-%d:set-%s',ep,layerId,cell2mat(set)));
                fprintf('Satrt Train/Test! epcoh-%d:layerIndex-%d:set-%s',ep,layerId,cell2mat(set));
                cd E:\MatConvNet-1.0-beta17\examples\imagenet-Xsparse;
                cnn_imagenetX_test(cell2mat(set),ep,layerId);
                fprintf('\nEnd! -----Test Done!!-----\n'); 
                toc(time);
            end
        end
        if calcJD
            for set = sets
                time = tic;
                fprintf('Satrt JD! epcoh-%d:layerIndex-%d',ep,layerId);
                cd E:\CNN1_FeatureEvluation;
                imagenet_Main2(ep,layerId,cell2mat(set));
                fprintf('\nEnd! -----JD Done!!-----\n'); 
                toc(time);
            end
        end
    end
end

