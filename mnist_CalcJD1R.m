function mnist_CalcJD1R(ep,layerId)
% clear all;
% close all;
% set(0,'DefaultFigureWindowStyle','normal');
% 

% ep = 0; layerId = 9; 
set = {'Train','Test'}; epoch = ep; layerIndex = layerId;

dataDir1 = 'E:\MatConvNet-1.0-beta17\examples\mnist\data\mnist';
dataDir2 = 'E:\MatConvNet-1.0-beta17\examples\mnist\data\lenet-test-jd1-1';
expDir = 'mnist-lenet-exp-jd1-1';

% 导入数据
dataName1 = sprintf('epoch-%d-layer-%d-%s.mat',epoch,layerIndex,set{1});
dataName2 = sprintf('epoch-%d-layer-%d-%s.mat',epoch,layerIndex,set{2});
if ~exist(expDir,'dir')
    mkdir(expDir);
end

imdb = load(fullfile(dataDir1,'imdb.mat'));
trainData = load(fullfile(dataDir2,dataName1));
testData = load(fullfile(dataDir2,dataName2));
trainData = single(trainData.result);
testData = single(testData.result);
trainLabel = imdb.images.labels(imdb.images.set==1);
testLabel = imdb.images.labels(imdb.images.set==3);
classNum = max(trainLabel);
clear imdb

% 数据格式整理
% trainData = squeeze(cell2mat(cellfun(@(x)cell2mat(x),trainData,'UniformOutput', false)));
% testData = squeeze(cell2mat(cellfun(@(x)cell2mat(x),testData,'UniformOutput', false)));
% trainLabelVec = full(ind2vec(double(trainLabel))); 
% testLabelVec = full(ind2vec(double(testLabel)));

% % 数据按类排序
% [trainLabel,Ind] = sort(trainLabel,'ascend');
% trainData = trainData(:,Ind);
% [testLabel,Ind] = sort(testLabel,'ascend');
% testData = testData(:,Ind);

% 数据保存
% data.trainData = trainData;
% data.trainLabel = trainLabel;
% data.testData = testData;
% data.testLabel = testLabel;
% save cifar10_Data.mat trainData trainLabel testData testLabel;

% 计算各类的pdf直方图--边缘分布
% 已假定各边缘分布相互独立，因此统计在所有类的每个维度方向上单独进行
% scaleTrain = max(max(trainData)) - min(min(trainData));
% scaleTest = max(max(testData)) - min(min(testData));
% scaleTrain = ep + 1;
% scaleTest = ep + 1;
scaleTrain = [-300,300];
scaleTest = [-300,300];
binWidth = 0.3;
% trainData = mapminmax(trainData,0,1);
% testData = mapminmax(testData,0,1);
trainDensities = cell(1,classNum);
testDensities = cell(1,classNum);
disp('Calculate Densities of Each Class ...');
for classIdx = 1:classNum
    trainDensities{classIdx} = calcVectorsHistogramR(trainData(:,trainLabel==classIdx),scaleTrain,binWidth);
    testDensities{classIdx} = calcVectorsHistogramR(testData(:,testLabel==classIdx),scaleTest,binWidth);
end
clear trainData testData trainLabel testLabel;

%计算任意2类的类间概率散度JD
trainJD = zeros(classNum,classNum);
testJD = zeros(classNum,classNum);
for i = 1: classNum
    for j = 2:classNum
        if i<j
            trainJD(i,j) = calcProbaDiverge(trainDensities{i},trainDensities{j});
            testJD(i,j) = calcProbaDiverge(testDensities{i},testDensities{j});
        end
    end
end
JD.train = sum(sum(triu(trainJD)))/nchoosek(classNum,2);
JD.test = sum(sum(triu(testJD)))/nchoosek(classNum,2);

% 保存数据
dataName = sprintf('layer-%d-epoch-%d-JD.mat',layerIndex,epoch);
save(fullfile(expDir,dataName),'JD');

