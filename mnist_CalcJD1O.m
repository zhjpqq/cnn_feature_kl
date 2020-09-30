function mnist_CalcJD1O(ep,layerId)
% clear all;
% close all;
% set(0,'DefaultFigureWindowStyle','normal');
% ��һ����0~1֮��

% ep = 0; layerId = 9; 
set = {'Train','Test'}; epoch = ep; layerIndex = layerId;

dataDir1 = 'E:\MatConvNet-1.0-beta17\examples\mnist\data\mnist';
dataDir2 = 'E:\MatConvNet-1.0-beta17\examples\mnist\data\lenet5-test-jd1';
expDir = 'mnist-lenet5-jd1';

% ��������
dataName1 = sprintf('epoch-%d-layer-%d-%s.mat',epoch,layerIndex,set{1});
dataName2 = sprintf('epoch-%d-layer-%d-%s.mat',epoch,layerIndex,set{2});
if ~exist(expDir,'dir')
    mkdir(expDir);
end

imdb = load(fullfile(dataDir1,'imdb.mat'));
trainData = load(fullfile(dataDir2,dataName1));
testData = load(fullfile(dataDir2,dataName2));
trainData = trainData.result;
testData = testData.result;
trainLabel = imdb.images.labels(imdb.images.set==1);
testLabel = imdb.images.labels(imdb.images.set==3);
classNum = max(trainLabel);
clear imdb

% ��������pdfֱ��ͼ--��Ե�ֲ�
% �Ѽٶ�����Ե�ֲ��໥���������ͳ�����������ÿ��ά�ȷ����ϵ�������
trainData = mapminmax(trainData,0,1);
testData = mapminmax(testData,0,1);
trainDensities = cell(1,classNum);
testDensities = cell(1,classNum);
disp('Calculate Densities of Each Class ...');
for classIdx = 1:classNum
    trainDensities{classIdx} = calcVectorsHistogramO(trainData(:,trainLabel==classIdx));
    testDensities{classIdx} = calcVectorsHistogramO(testData(:,testLabel==classIdx));
end
clear trainData testData trainLabel testLabel;

%��������2���������ɢ��JD
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

% ��������
dataName = sprintf('layer-%d-epoch-%d-JD.mat',layerIndex,epoch);
save(fullfile(expDir,dataName),'JD');

