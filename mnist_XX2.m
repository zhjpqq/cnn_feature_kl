% mnist-lenet
% layer|   0|    1|    2|    3|    4|    5|    6|    7|    8|    9|     10|
% type|input| conv| relu|mpool| conv| relu|mpool| conv| relu| conv|softmxl|
% name|  n/a|conv1|relu1|pool1|conv2|relu2|pool2|conv3|relu3|conv4|    sml|

% �������batch�µ���������
% ������������ɢ�ȵ�������

calcFeature = false;
calcJd = true;

%��������ͼ
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

%����ɢ��
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
                %��0��batch�͵�599��batch���ڱ�epoch��ѭ��
                epn = ep;
                batchIdn = batchId + 20;
            else
                %���ﵱǰepoch�ĵ�600��batch�����п�epcohѭ��
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
            %������һ�� layerIndex
            break;
        end        
    end
end
end
