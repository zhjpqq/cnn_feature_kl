function JD = calcProbaDiverge(Pxwa,Pxwb,varargin)
% ��������ĸ��ʷֲ� P(x|Wa)��P(x|Wb).
% ���������ɢ�� JD =��[(p(x|Wa)-p(x|Wb))ln(p(x|Wa)/p(x|Wb))]dx, x��X.
% �Ը���ֱ��ͼPDF�����൱�����ۼƷֲ�F(X).
% �μ���ģʽʶ�𡷱� ����ڶ��棬8.2�ڣ���ʽ8-17.

assert(all(size(Pxwa)==size(Pxwb)),'error:the two PDF cannot match!');

% Pxwa = double(Pxwa);  Pxwb = double(Pxwb);
% [xDim, xNum] = size(Pxwa);
% JD = zeros(xDim,xNum);
% for d = 1:xDim       %��������ά��
%     for n = 1:xNum   %��������bins 
%         JD(d,n) = (Pxwa(d,n)-Pxwb(d,n))*log10(Pxwa(d,n)/Pxwb(d,n));
%     end
% end
% JD = sum(sum(JD));

% �ĳ�GPU����
if nargin==2
    JD = (Pxwa - Pxwb).*log10(Pxwa./Pxwb);
    JD =sum(sum(JD));
else
    Pxwa = gpuArray(Pxwa);
    Pxwb = gpuArray(Pxwb);
    JD = (Pxwa - Pxwb).*log10(Pxwa./Pxwb);
    JD = gather(sum(sum(JD)));
end

