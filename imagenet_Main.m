function imagenet_Main(ep,layerId)
% clear all;
% close all;
% set(0,'DefaultFigureWindowStyle','normal');

% 导入数据
set = {'Val','Train'}; epoch = ep; layerIndex = layerId;
dataDir1 = 'E:\MatConvNet-1.0-beta17\examples\imagenet-Xsparse\data';
dataDir2 = 'E:\MatConvNet-1.0-beta17\examples\imagenet-Xsparse\data\alexnet-test';
dataName1 = sprintf('epoch-%d-layer-%d-%s.mat',epoch,layerIndex,set{1});
dataName2 = sprintf('epoch-%d-layer-%d-%s.mat',epoch,layerIndex,set{2});
expDir = 'imagenet-alexnet-exp';
if ~exist('expDir','dir')
    mkdir(expDir);
end

tic;
imdb = load(fullfile(dataDir1,'oimdb.mat'));
trainData = load(fullfile(dataDir2,dataName1));
valData = load(fullfile(dataDir2,dataName2));
trainData = trainData.result;
valData = valData.result;
trainLabel = imdb.images.label(imdb.images.set==1);
valLabel = imdb.images.label(imdb.images.set==2);
classNum = max(trainLabel);
clear imdb dataDir1 dataDir2 dataName1 dataName2；

% 数据格式整理
trainData = squeeze(cell2mat(cellfun(@(x)cell2mat(x),trainData,'UniformOutput', false)));
valData = squeeze(cell2mat(cellfun(@(x)cell2mat(x),valData,'UniformOutput', false)));
% trainLabelVec = full(ind2vec(double(trainLabel))); 
% testLabelVec = full(ind2vec(double(testLabel)));
disp('trainData & valData prepare done! ....');
toc;

% 数据按类排序
% [trainLabel,Ind] = sort(trainLabel,'ascend');
% trainData = trainData(:,Ind);
% [valLabel,Ind] = sort(valLabel,'ascend');
% valData = valData(:,Ind);

% 计算各类在每一维度上的pdf直方图--边缘分布
% 已假定各边缘分布相互独立，因此统计在所有类的每个维度方向上单独进行
tic;
trainData = mapminmax(trainData,0,1);
valData = mapminmax(valData,0,1);
trainDensities = cell(1,classNum);
valDensities = cell(1,classNum);
for classIdx = 1:classNum
    trainDensities{classIdx} = calcVectorsHistogram(trainData(:,trainLabel==classIdx));
    valDensities{classIdx} = calcVectorsHistogram(valData(:,valLabel==classIdx));
end
clear trainData valData trainLabel valLabel;
disp('calculate trainDensities & valDensities done! ....');
toc;

% 第i类在第dim维上的pdf
% i=1; dim=1;
% plot(trainDensities{i}(dim,:));

%计算任意2类的类间概率散度JD
tic;
trainJD = zeros(classNum,classNum);
valJD = zeros(classNum,classNum);
for i = 1: classNum
    for j = 1:classNum
        if i<j
            trainJD(i,j) = calcProbaDiverge(trainDensities{i},trainDensities{j});
            valJD(i,j) = calcProbaDiverge(valDensities{i},valDensities{j});
        end
    end
end
JD.train = sum(sum(triu(trainJD)))/nchoosek(classNum,2);
JD.val = sum(sum(triu(valJD)))/nchoosek(classNum,2);
disp('calculate JD.train & JD.test done! ....');
toc;

% 保存数据
tic;
dataName = sprintf('layer-%d-epoch-%d-JD.mat',layerIndex,epoch);
save(fullfile(expDir,dataName),'JD');
disp('save JD done! ....');
toc;
