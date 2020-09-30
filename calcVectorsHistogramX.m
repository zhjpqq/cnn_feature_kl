function densities = calcVectorsHistogramX(vectorsMat)
% ���������������ɵľ���vectorsMat��������ÿ������ά�ȵĸ����ܶȷֲ�desities
% ���룺vectorsMat m��n  m:featuDim  n:nSamples  
% �����densities m��binSize  featuDim��binSize���ȵĸ����ܶȷֲ�
% **** Ϊȷ��pdf������ֵ�����Ƚ�densities��ʼ��Ϊ���ȷֲ�
% **** �������ȹ�һ������ÿһ�е�ֵ��0,1����

[featuDim,nSamples]=size(vectorsMat);

% ��ʼ��ֱ��ͼ
binSize = 100;
binWidth = 1.0/(binSize-1);
densities = ones(featuDim,binSize);   

% ͳ��ֱ��ͼ
for m = 1:featuDim    %��������
    for n=1:nSamples  %��������
        idx = floor(vectorsMat(m,n)/binWidth)+1;  % 1��idx��100
        if idx < 1    %��Խ��ֵ���нضϴ���    
            idx = 1; disp('idx < 1');
        elseif idx >100;
            idx = 100; disp('idx > 100');
        end
        densities(m,idx) = densities(m,idx)+1;    % bin[idx]+1
    end
end

% ��һ��ֱ��ͼ
for m = 1:featuDim
    densities(m,:) = densities(m,:)./(sum(densities(m,:),2)+binSize);
end 
densities = single(densities);



