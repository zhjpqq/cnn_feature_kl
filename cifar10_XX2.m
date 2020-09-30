
% cifar-lenet
% layer|   0|    1|      2|   3|    4|   5|      6|    7|   8|      9|   10|  11|  12|     13|
% type|input| conv|  mpool|relu| conv|relu|  apool| conv|relu|  apool| conv|relu|conv|softmxl|
% name|  n/a|     |       |    |     |    |       |     |    |       |     |    |    |       |

% ���ȼ���������batch�µ���������
% �ټ�������batch�������

calcFeature = true;
calcJd = false;

%��������ͼ
if calcFeature
epoch = 1;
batchIndex = 0:20:500;
layerIndex = 12:-1:10;
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
                cd E:\MatConvNet-1.0-beta17\examples\cifar;
                cnn_cifarX_test2(cell2mat(set),ep,layerId,batchId);
                fprintf('\nEnd! -----Train/Test Done!!-----\n'); 
                toc(time);
            end
        end
    end
end
end

%����ǰ��batch��ɢ��
if calcJd
epoch = 1;
batchIndex = 0:20:500;
layerIndex = 12:-1:0;
sets = {'Train','Test'};
overep = 0;
for layerId = layerIndex
    for ep = epoch
        for batchId = batchIndex
            if ep>1 && batchId == 0
                %>1 ��epoch�����������batch
                continue;   
            end
            time = tic;
            fprintf('Satrt JD! epcoh-%d:layerIndex-%d:batchId-%d\n',ep,layerId,batchId);
            cd E:\CNN1_FeatureEvluation;
            if batchId ==0 || ~(mod(batchId,500)==0)
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
            cifar10_Main2(layerId,ep,epn,batchId,batchIdn);
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
