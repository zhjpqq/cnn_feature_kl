function JD = calcProbaDiverge(Pxwa,Pxwb,varargin)
% 给定两类的概率分布 P(x|Wa)、P(x|Wb).
% 计算两类的散度 JD =∫[(p(x|Wa)-p(x|Wb))ln(p(x|Wa)/p(x|Wb))]dx, x∈X.
% 对概率直方图PDF积分相当于求累计分布F(X).
% 参见《模式识别》边 肇齐第二版，8.2节，公式8-17.

assert(all(size(Pxwa)==size(Pxwb)),'error:the two PDF cannot match!');

% Pxwa = double(Pxwa);  Pxwb = double(Pxwb);
% [xDim, xNum] = size(Pxwa);
% JD = zeros(xDim,xNum);
% for d = 1:xDim       %遍历特征维度
%     for n = 1:xNum   %遍历特征bins 
%         JD(d,n) = (Pxwa(d,n)-Pxwb(d,n))*log10(Pxwa(d,n)/Pxwb(d,n));
%     end
% end
% JD = sum(sum(JD));

% 改成GPU计算
if nargin==2
    JD = (Pxwa - Pxwb).*log10(Pxwa./Pxwb);
    JD =sum(sum(JD));
else
    Pxwa = gpuArray(Pxwa);
    Pxwb = gpuArray(Pxwb);
    JD = (Pxwa - Pxwb).*log10(Pxwa./Pxwb);
    JD = gather(sum(sum(JD)));
end

