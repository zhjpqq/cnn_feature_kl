function densities = calcVectorsHistogramY(vectorsMat,scale)
% 传入特征向量构成的矩阵vectorsMat，计算其每个特征维度的概率密度分布desities
% 输入：vectorsMat m×n  m:featuDim  n:nSamples 
% 输入：scale: 特征向量的动态范围：maxValue - minValue
% 输出：densities m×binSize  featuDim个binSize长度的概率密度分布
% **** 为确保pdf中无零值，事先将densities初始化为均匀分布
% **** vectorsMat应 0≤值≤1
% 按迭代次数放大统计区间，效果差

[featuDim,nSamples]=size(vectorsMat);

% 截断到0-1范围内
if any(min(vectorsMat,[],2)<0)
    fprintf('小于0截断--%4f---',numel(find(vectorsMat<0))/(featuDim*nSamples));
    pause(30);
    vectorsMat(find(vectorsMat<0)) = 0;
end
if any(max(vectorsMat,[],2)>1)
    fprintf('大于1截断--%4f---',numel(find(vectorsMat>1))/(featuDim*nSamples));
    pause(30);
    vectorsMat(find(vectorsMat>1)) = 1;
end

% 初始化直方图
binSize = floor(100*scale);
binWidth = 1.0/(binSize-1);
densities = ones(featuDim,binSize);   

% 统计直方图
for m = 1:featuDim    %遍历特征
    for n=1:nSamples  %遍历样本
        idx = floor(vectorsMat(m,n)/binWidth)+1;  % 1≤idx≤100
        densities(m,idx) = densities(m,idx)+1;    % bin[idx]+1
    end
end

% 归一化直方图
for m = 1:featuDim
    densities(m,:) = densities(m,:)./(sum(densities(m,:),2)+binSize);
end 
densities = single(densities);



