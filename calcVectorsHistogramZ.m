function densities = calcVectorsHistogramZ(vectorsMat,scale,binSize)
% 传入特征向量构成的矩阵vectorsMat，计算其每个特征维度的概率密度分布desities
% 输入：vectorsMat m×n  m:featuDim  n:nSamples 
% 输入：scale: 特征向量的取值范围：(minValue -> maxValue)
% 输入：size：直方图的尺寸
% 输出：densities m×binSize  featuDim个binSize长度的概率密度分布
% **** 为确保pdf中无零值，事先将densities初始化为均匀分布
% **** vectorsMat 无需归一化，给定范围scale和区间数量binSize

[featuDim,nSamples]=size(vectorsMat);
minValue = scale(1);
maxValue = scale(2);

% 截断到min-max范围内
if any(min(vectorsMat,[],2)<minValue)
    fprintf('小于0截断--%4f---',numel(find(vectorsMat<minValue))/(featuDim*nSamples));
    pause(30);
    vectorsMat(find(vectorsMat<minValue)) = minValue;
end
if any(max(vectorsMat,[],2)>maxValue)
    fprintf('大于1截断--%4f---',numel(find(vectorsMat>maxValue))/(featuDim*nSamples));
    pause(30);
    vectorsMat(find(vectorsMat>maxValue)) = maxValue;
end

% 初始化直方图
binSize = binSize;
binWidth = (maxValue-minValue)/(binSize-1);
densities = ones(featuDim,binSize);   

% 统计直方图
for m = 1:featuDim    %遍历特征
    for n=1:nSamples  %遍历样本
        idx = floor((vectorsMat(m,n)-minValue)/binWidth)+1;  % 1≤idx≤500
        densities(m,idx) = densities(m,idx)+1;    % bin[idx]+1
    end
end

% 归一化直方图
for m = 1:featuDim
    densities(m,:) = densities(m,:)./(sum(densities(m,:),2)+binSize);
end 
% densities = single(densities);


