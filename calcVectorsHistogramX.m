function densities = calcVectorsHistogramX(vectorsMat)
% 传入特征向量构成的矩阵vectorsMat，计算其每个特征维度的概率密度分布desities
% 输入：vectorsMat m×n  m:featuDim  n:nSamples  
% 输出：densities m×binSize  featuDim个binSize长度的概率密度分布
% **** 为确保pdf中无零值，事先将densities初始化为均匀分布
% **** 无需事先归一化矩阵每一行的值到0,1区间

[featuDim,nSamples]=size(vectorsMat);

% 初始化直方图
binSize = 100;
binWidth = 1.0/(binSize-1);
densities = ones(featuDim,binSize);   

% 统计直方图
for m = 1:featuDim    %遍历特征
    for n=1:nSamples  %遍历样本
        idx = floor(vectorsMat(m,n)/binWidth)+1;  % 1≤idx≤100
        if idx < 1    %对越界值进行截断处理    
            idx = 1; disp('idx < 1');
        elseif idx >100;
            idx = 100; disp('idx > 100');
        end
        densities(m,idx) = densities(m,idx)+1;    % bin[idx]+1
    end
end

% 归一化直方图
for m = 1:featuDim
    densities(m,:) = densities(m,:)./(sum(densities(m,:),2)+binSize);
end 
densities = single(densities);



