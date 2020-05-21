#include<stdio.h>
#include<stdlib.h>
#define TPB 8
#define W 4
#define H 4
#define TX 1
#define TY 1

int N=H*W;


__device__ float distance(float x1, float x2){
 return sqrt ((x2-x1)*(x2-x1));
}

__global__ void distanceKernel(float *d_out, float *d_in, float ref, int w){
	const int c=blockIdx.x*blockDim.x+threadIdx.x;
	const int r=blockIdx.y*blockDim.y+threadIdx.y;
	const int i=r*w+c;
	
	const float x=d_in[i];
	d_out[i]=distance(x, ref);

	printf("c=%d, r=%d, i=%d: the distance between %f to %f is %f. \n", c, r, i, ref, x, d_out[i]); ////
}

void distanceArray(float *out, float *in, float ref, int len){
	float *d_in=0;
	float *d_out=0; ////

	cudaMalloc(&d_in, len*sizeof(float));
	cudaMalloc(&d_out, len*sizeof(float));

	cudaMemcpy(d_in, in, len*sizeof(float), cudaMemcpyHostToDevice);

	const dim3 blockSize(TX, TY);
	const int bx=(W+TX-1)/TX;
	const int by=(W+TY-1)/TY;
	const dim3 gridSize=dim3(bx, by);

	distanceKernel<<<gridSize, blockSize>>>(d_out, d_in, ref, W);

	cudaMemcpy(out, d_out, len*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);
}

float scale(int i, int n){
 return ((float) i)/(n-1);
}

int main(){
	const float ref=0.5f;
	
	float *in=(float*) calloc(N,sizeof(float));
	float *out=(float*) calloc(N, sizeof(float));

	for(int i=0; i<N; ++i){
		in[i]=scale(i,N); //
	}

	distanceArray(out, in, ref, N);

	printf("______________________________ \n");

	for(int j=0; j<N; ++j){
		printf("The distance, printed from the host, between %f to %f is %f. \n", ref, in[j], out[j]);
	}

	free(in);
	free(out);

	return 0;
}