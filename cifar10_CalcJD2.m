function cifar10_Main2(layerId,ep,epn,batchId,batchIdn)
% clear all;
% cifar10_Main2(layerId,ep,epn,batchId,batchIdn);
% close all;
% ��������epoch��next epoch�� batch������ next batch
% �Ե���������������������JD-Batch���ֱ�ΪTrain��Test����ǰ��2��batch�����ݽ��м���

% ��������
% layerId = 1; ep = 1; epn = ep; batchId = 0;  batchIdn = 20;
set = {'Train','Test'}; 
layerIndex = layerId; 
epochP = ep;  epochN = epn;
batchIndexP = batchId;  batchIndexN = batchIdn; 

dataDir1 = 'E:\MatConvNet-1.0-beta17\examples\cifar\data\cifar-lenet-simplenn';
dataDir2 = 'E:\MatConvNet-1.0-beta17\examples\cifar\data\lenet-test';
dataName1 = sprintf('epoch-%d-layer-%d-batch-%d-%s.mat',epochP,layerIndex,batchIndexP,set{1});
dataName3 = sprintf('epoch-%d-layer-%d-batch-%d-%s.mat',epochP,layerIndex,batchIndexN,set{1});
dataName2 = sprintf('epoch-%d-layer-%d-batch-%d-%s.mat',epochN,layerIndex,batchIndexP,set{2});
dataName4 = sprintf('epoch-%d-layer-%d-batch-%d-%s.mat',epochN,layerIndex,batchIndexN,set{2});
expDir = 'cifar-lenet-exp';
if ~exist(expDir,'dir')
    mkdir(expDir);
end

% load label 
imdb = load(fullfile(dataDir1,'imdb.mat'));
trainLabel = imdb.images.labels(imdb.images.set==1);
testLabel = imdb.images.labels(imdb.images.set==3);
classNum = max(trainLabel);
clear imdb;
% load prior batch 
trainData1 = load(fullfile(dataDir2,dataName1));
testData1 = load(fullfile(dataDir2,dataName2));
trainData1 = trainData1.result;
testData1 = testData1.result;
% load net batch
trainData2 = load(fullfile(dataDir2,dataName3));
testData2 = load(fullfile(dataDir2,dataName4));
trainData2 = trainData2.result;
testData2 = testData2.result;
clear dataDir1 dataDir2 dataName1 dataName2 dataName3 dataName4��

% ���ݸ�ʽ����
trainData1 = squeeze(cell2mat(cellfun(@(x)cell2mat(x),trainData1,'UniformOutput', false)));
testData1 = squeeze(cell2mat(cellfun(@(x)cell2mat(x),testData1,'UniformOutput', false)));
trainData2 = squeeze(cell2mat(cellfun(@(x)cell2mat(x),trainData2,'UniformOutput', false)));
testData2 = squeeze(cell2mat(cellfun(@(x)cell2mat(x),testData2,'UniformOutput', false)));
% trainLabelVec = full(ind2vec(double(trainLabel))); 
% testLabelVec = full(ind2vec(double(testLabel)));

% % ���ݰ�������
% [trainLabel,Ind] = sort(trainLabel,'ascend');
% trainData = trainData(:,Ind);
% [testLabel,Ind] = sort(testLabel,'ascend');
% testData = testData(:,Ind);

% ��������pdfֱ��ͼ--��Ե�ֲ�
% �Ѽٶ�����Ե�ֲ��໥���������ͳ�����������ÿ��ά�ȷ����ϵ�������
% prior bacth
tic;
disp('Calculate Densities of Each Class @ prior Batch ...');
trainData1 = mapminmax(trainData1,0,1);
testData1 = mapminmax(testData1,0,1);
trainDensities1 = cell(1,classNum);
testDensities1 = cell(1,classNum);
for classIdx = 1:classNum
    trainDensities1{classIdx} = calcVectorsHistogram(trainData1(:,trainLabel==classIdx));
    testDensities1{classIdx} = calcVectorsHistogram(testData1(:,testLabel==classIdx));
end
clear trainData1 testData1;
toc;
% next batch
tic;
disp('Calculate Densities of Each Class @ next Batch ...');
trainData2 = mapminmax(trainData2,0,1);
testData2 = mapminmax(testData2,0,1);
trainDensities2 = cell(1,classNum);
testDensities2 = cell(1,classNum);
for classIdx = 1:classNum
    trainDensities2{classIdx} = calcVectorsHistogram(trainData2(:,trainLabel==classIdx));
    testDensities2{classIdx} = calcVectorsHistogram(testData2(:,testLabel==classIdx));
end
clear trainData2 testData2;
toc;

%�������ⵥ���bacth�����ɢ��JD
trainJD = zeros(1,classNum);
testJD = zeros(1,classNum);
for i = 1: classNum
    trainJD(i) = calcProbaDiverge(trainDensities1{i},trainDensities2{i});
    testJD(i) = calcProbaDiverge(testDensities1{i},testDensities2{i});
end
JD.train = sum(trainJD)/classNum;
JD.test = sum(testJD)/classNum;

% ��������
dataName = sprintf('layer-%d-epoch-%d-batch-%d-JD.mat',layerIndex,epochP,batchIndexP);
save(fullfile(expDir,dataName),'JD');

