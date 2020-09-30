function imagenet_Main2(ep,layerId,sets)
% clear all;
% close all;
% set(0,'DefaultFigureWindowStyle','normal');

% set = 'Val'; epoch = 1000; layerIndex = 20;
set = sets; epoch = ep; layerIndex = layerId;
imdbDir = 'E:\MatConvNet-1.0-beta17\examples\imagenet-Xsparse\data';
resDir = 'E:\MatConvNet-1.0-beta17\examples\imagenet-Xsparse\data\alexnet-test';
expDir = 'imagenet-alexnet-exp';
if ~exist(expDir,'dir')
    mkdir(expDir);
end

% ��������
tic;
imdb = load(fullfile(imdbDir,'oimdb.mat'));
setIdx = 0;
switch lower(set)
    case 'train'
        setIdx = 1;
    case 'val'
        setIdx = 2;
    case 'test'
        setIdx = 3;
    case 'others'
        error('Wrong Set Idx��');
end
setLabel = imdb.images.label(imdb.images.set==setIdx);
classNum = max(setLabel);
resName =  sprintf('epoch-%d-layer-%d-%s.mat',epoch,layerIndex,set);
resData = load(fullfile(resDir,resName));
resData = resData.result;
clear imdb imdbDir resDir resName��

% ���ݸ�ʽ����
resData = squeeze(cell2mat(cellfun(@(x)cell2mat(x),resData,'UniformOutput',false)));
% trainData = squeeze(cell2mat(cellfun(@(x)cell2mat(x),trainData,'UniformOutput', false)));
% valData = squeeze(cell2mat(cellfun(@(x)cell2mat(x),valData,'UniformOutput', false)));
% trainLabelVec = full(ind2vec(double(trainLabel))); 
% testLabelVec = full(ind2vec(double(testLabel)));
disp('resData & setLabel prepare done! ....');
toc;

% ���ݰ�������
% [trainLabel,Ind] = sort(trainLabel,'ascend');
% trainData = trainData(:,Ind);
% [valLabel,Ind] = sort(valLabel,'ascend');
% valData = valData(:,Ind);

% ���������ÿһά���ϵ�pdfֱ��ͼ--��Ե�ֲ�
% �Ѽٶ�����Ե�ֲ��໥���������ͳ�����������ÿ��ά�ȷ����ϵ�������
tic;
resData = mapminmax(resData,0,1);
densities = cell(1,classNum);
for classIdx = 1:classNum
    densities{classIdx} = calcVectorsHistogram(resData(:,setLabel==classIdx));
end
clear resData setLabel;
disp('calculate Densities done! ....');
toc;

% ��i���ڵ�dimά�ϵ�pdf
% i=1; dim=1;
% plot(trainDensities{i}(dim,:));

%��������2���������ɢ��JD
tic;
JD = zeros(classNum,classNum);
for i = 1: classNum
    for j = 1:classNum
        if i<j
            JD(i,j) = calcProbaDiverge(densities{i},densities{j});
        end
    end
end
JD = sum(sum(triu(JD)))/nchoosek(classNum,2);
disp('calculate JD done! ....');
toc;

% ��������
tic;
dataName = sprintf('layer-%d-epoch-%d-%s-JD.mat',layerIndex,epoch,set);
save(fullfile(expDir,dataName),'JD');
disp('save JD done! ....');
toc;
