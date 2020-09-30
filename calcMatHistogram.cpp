#include "mex.h"
#include "math.h"
#include "float.h"

static size_t m=0; 
static size_t n=0;

static int binSize = 100;
static double binWidth = 1.0 / (binSize - 1);

double* calcMatHist(float* a)
{
	size_t mDim = m;
	size_t nSamples = n;

	double* featurePdf = new double[mDim,binSize];
	for (size_t i = 0; i < mDim; i++)
	{
		for (size_t j = 0; j < binSize; j++)
		{
			featurePdf[i, j] = 1;
		}
	}

	int idx;
	for (size_t i = 0; i < mDim; i++)
	{
		for (size_t j = 0; j < nSamples; j++)
		{
			idx = floor(a[i, j] / binWidth);
			featurePdf[i, idx] = featurePdf[i, idx] + 1;
		}
	}
	
	for(size_t i = 0; i < mDim; i++)
		for (size_t j = 0; j < binSize; j++)
			featurePdf[i, j] = featurePdf[i, j] / (nSamples + binSize);

	return featurePdf;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	float* featuVec;
	m = mxGetM(prhs[0]);
	n = mxGetN(prhs[0]);
	featuVec = (float *)mxGetPr(prhs[0]);

	double* featuPdf = calcMatHist(featuVec);

	plhs[0] = mxCreateDoubleMatrix(m, binSize, mxREAL);
	double* p = mxGetPr(plhs[0]);

	for (size_t i = 0; i < m;i++)
		for (size_t j = 0; j < binSize; j++)
			p[i, j] = featuPdf[i, j];

	delete [] featuPdf;
	featuPdf = NULL;
// 	delete [] featuVec;
// 	featuVec = NULL;
//  传入变量的指针不能删，否则崩溃
}