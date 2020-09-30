function densities = calcVectorsHistogramZ(vectorsMat,scale,binSize)
% ���������������ɵľ���vectorsMat��������ÿ������ά�ȵĸ����ܶȷֲ�desities
% ���룺vectorsMat m��n  m:featuDim  n:nSamples 
% ���룺scale: ����������ȡֵ��Χ��(minValue -> maxValue)
% ���룺size��ֱ��ͼ�ĳߴ�
% �����densities m��binSize  featuDim��binSize���ȵĸ����ܶȷֲ�
% **** Ϊȷ��pdf������ֵ�����Ƚ�densities��ʼ��Ϊ���ȷֲ�
% **** vectorsMat �����һ����������Χscale����������binSize

[featuDim,nSamples]=size(vectorsMat);
minValue = scale(1);
maxValue = scale(2);

% �ضϵ�min-max��Χ��
if any(min(vectorsMat,[],2)<minValue)
    fprintf('С��0�ض�--%4f---',numel(find(vectorsMat<minValue))/(featuDim*nSamples));
    pause(30);
    vectorsMat(find(vectorsMat<minValue)) = minValue;
end
if any(max(vectorsMat,[],2)>maxValue)
    fprintf('����1�ض�--%4f---',numel(find(vectorsMat>maxValue))/(featuDim*nSamples));
    pause(30);
    vectorsMat(find(vectorsMat>maxValue)) = maxValue;
end

% ��ʼ��ֱ��ͼ
binSize = binSize;
binWidth = (maxValue-minValue)/(binSize-1);
densities = ones(featuDim,binSize);   

% ͳ��ֱ��ͼ
for m = 1:featuDim    %��������
    for n=1:nSamples  %��������
        idx = floor((vectorsMat(m,n)-minValue)/binWidth)+1;  % 1��idx��500
        densities(m,idx) = densities(m,idx)+1;    % bin[idx]+1
    end
end

% ��һ��ֱ��ͼ
for m = 1:featuDim
    densities(m,:) = densities(m,:)./(sum(densities(m,:),2)+binSize);
end 
% densities = single(densities);


