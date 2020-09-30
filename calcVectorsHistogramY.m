function densities = calcVectorsHistogramY(vectorsMat,scale)
% ���������������ɵľ���vectorsMat��������ÿ������ά�ȵĸ����ܶȷֲ�desities
% ���룺vectorsMat m��n  m:featuDim  n:nSamples 
% ���룺scale: ���������Ķ�̬��Χ��maxValue - minValue
% �����densities m��binSize  featuDim��binSize���ȵĸ����ܶȷֲ�
% **** Ϊȷ��pdf������ֵ�����Ƚ�densities��ʼ��Ϊ���ȷֲ�
% **** vectorsMatӦ 0��ֵ��1
% �����������Ŵ�ͳ�����䣬Ч����

[featuDim,nSamples]=size(vectorsMat);

% �ضϵ�0-1��Χ��
if any(min(vectorsMat,[],2)<0)
    fprintf('С��0�ض�--%4f---',numel(find(vectorsMat<0))/(featuDim*nSamples));
    pause(30);
    vectorsMat(find(vectorsMat<0)) = 0;
end
if any(max(vectorsMat,[],2)>1)
    fprintf('����1�ض�--%4f---',numel(find(vectorsMat>1))/(featuDim*nSamples));
    pause(30);
    vectorsMat(find(vectorsMat>1)) = 1;
end

% ��ʼ��ֱ��ͼ
binSize = floor(100*scale);
binWidth = 1.0/(binSize-1);
densities = ones(featuDim,binSize);   

% ͳ��ֱ��ͼ
for m = 1:featuDim    %��������
    for n=1:nSamples  %��������
        idx = floor(vectorsMat(m,n)/binWidth)+1;  % 1��idx��100
        densities(m,idx) = densities(m,idx)+1;    % bin[idx]+1
    end
end

% ��һ��ֱ��ͼ
for m = 1:featuDim
    densities(m,:) = densities(m,:)./(sum(densities(m,:),2)+binSize);
end 
densities = single(densities);



