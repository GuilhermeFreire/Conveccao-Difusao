#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define PRECISION 0.00001
#define uN 5.0
#define uS 5.0
#define uW 0.0
#define uE 10.0
#define TAM_BLOCO 1024

double intervaloX, intervaloY;
double denominador1, denominador2;
double *h_m, *d_m;
int divX, divY, laps;

//Essas sao as variaveis globais da GPU
__device__ double d_intervaloX;
__device__ double d_intervaloY;
__device__ double d_denominador1;
__device__ double d_denominador2;
__device__ int d_divX;
__device__ int d_divY;

__device__ double a(int i, int j){
	double x = i*d_intervaloX;
	double y = i*d_intervaloY;
	return 500 * x * (1 - x) * (0.5 - y);
}

__device__ double b(int i, int j){
	double x = i*d_intervaloX;
	double y = i*d_intervaloY;
	return 500 * y * (y - 1) * (x - 0.5);
}

__device__ double n(int i, int j){
	return (2.0 - d_intervaloX * b(i,j))/d_denominador2;
}
__device__ double s(int i, int j){
	return (2.0 + d_intervaloX * b(i,j))/d_denominador2;
}
__device__ double e(int i, int j){
	return (2.0 - d_intervaloX * a(i,j))/d_denominador1;
}
__device__ double w(int i, int j){
	return (2.0 + d_intervaloX * a(i,j))/d_denominador1;
}

__device__ double malhaGPU(double *matriz, int i, int j){
	i++;
	j++;
	// //OBS: Casos de canto não importam, pois nunca serão usados no problema.
	// //e.g. u(0,0) ou u(1,0) nunca serão usados no problema então tanto faz
	// //o valor retornado.
	if(j <= 0){
	// 	//uW foi escolhido para representar o uO do trabalho. 
	// 	//W vem de 'West', visto que todas as variáveis exceto essa foram
	// 	//nomeadas de acordo com sua cardinalidade em inglês
		return uW;
	}
	if(j >= d_divX + 2){
		return uE;
	}
	if(i <= 0){
		return uN;
	}
	if(i >= d_divY + 2){
		return uS;
	}
	i--;
	j--;
	return matriz[i * d_divX + j];
}

__device__ double u(double *matriz, int i, int j){
	// if(i == 0){
	// 	return uW;
	// }
	// if(i == divX + 2){
	// 	return uE;
	// }
	// if(j == 0){
	// 	return uN;
	// }
	// if(j == divX + 2){
	// 	return uS;
	// }
	// printf("u(%d, %d) = %lf * %lf + %lf * %lf + %lf * %lf + %lf * %lf\n", i, j, w(i,j), malha(i, j-1), e(i,j), malha(i, j+1), s(i,j), malha(i-1, j), n(i,j), malha(i+1, j));
	return w(i,j)*malhaGPU(matriz, i, j-1) + e(i,j)*malhaGPU(matriz, i, j+1) + s(i,j)*malhaGPU(matriz, i-1, j) + n(i,j)*malhaGPU(matriz, i+1, j);
}

__global__ void calculoAzul(double *matriz){
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j;

	i = 2*tidX/d_divX;
	j = 2*tidX%d_divY;

	matriz[i * (d_divX + 2) + j] = u(matriz, i, j);
}


__global__ void calculoVermelho(double *matriz){
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j;

	i = 2*tidX/d_divX;
	j = 2*tidX%d_divY + 1;

	matriz[i * (d_divX + 2) + j] = u(matriz, i, j);

}

double malhaCPU(double *matriz, int i, int j){
	i++;
	j++;
	// //OBS: Casos de canto não importam, pois nunca serão usados no problema.
	// //e.g. u(0,0) ou u(1,0) nunca serão usados no problema então tanto faz
	// //o valor retornado.
	if(j <= 0){
	// 	//uW foi escolhido para representar o uO do trabalho. 
	// 	//W vem de 'West', visto que todas as variáveis exceto essa foram
	// 	//nomeadas de acordo com sua cardinalidade em inglês
		return uW;
	}
	if(j >= divX + 2){
		return uE;
	}
	if(i <= 0){
		return uN;
	}
	if(i >= divY + 2){
		return uS;
	}
	i--;
	j--;
	return h_m[i * divX + j];
}

void printM(){
	int i = 0, j = 0;
	for(i = 0; i < divX + 2; i++){
		printf("%lf ", malhaCPU(h_m, 0 ,i));
	}
	printf("\n");

	for(i = 0; i < divX; i++){
		for(j = 0; j < divY; j++){
			if(j == 0){
				printf("%lf ", malhaCPU(h_m ,i ,j-1));
			}

			printf("%lf ", h_m[i* divX + j]);
			
			if(j == divY - 1){
				printf("%lf ", malhaCPU(h_m ,i ,j + 1));
			}
		}
		printf("\n");
	}

	for(i = 0; i < divX + 2; i++){
		printf("%lf ", malhaCPU(h_m, divX + 1,i));
	}
	printf("\n");
}


void printMat(){
	int i, j;
	for(i = 0; i < divX + 2; i++){
		for(j = 0; j < divY + 2; j++){
			printf("%lf", h_m[i*(divX + 2) + j]);
			if(j != divY + 1) printf(" ");
		}
		if(i != divX + 1)
			printf("\n");
	}
	printf("\n");
}

void setupM(){
	int i;
	for(i = 0; i < divY + 2; i++){
		h_m[i * (divX + 2)] = uW;
		h_m[i*(divX + 2) + divY + 1] = uE;
	}
	for(i = 0; i < divX + 2; i++){
		h_m[i] = uN;
		h_m[(divX + 1)*(divX + 2) + i] = uS;
	}
}

int main(int argc, char** argv){

	int laps = 0;
	int i;
	int threadsAzuis;
	int threadsVermelhas;

	if(argc < 2){
		printf("Número incorreto de parâmetros:\n");
		printf("Número de divisões faltando:\n");
		printf("\tPara valores iguais: %s <número de divisões> <Quantidade de iteracoes>\n", argv[0]);
		printf("\tPara valores diferentes: %s <divisões em X> <divisões em Y> <Quantidade de iteracoes>\n", argv[0]);
		exit(-1);
	}

	divX = atoi(argv[1]);
	divY = (argc > 2)? atoi(argv[2]): divX;
	laps = (argc > 3)? atoi(argv[3]): 1000;

	intervaloX = 1.0/(divX + 1);
	intervaloY = 1.0/(divY + 1);

	denominador1 = 4*(1 + ((intervaloX*intervaloX)/(intervaloY*intervaloY)));
	denominador2 = 4*(1 + ((intervaloY*intervaloY)/(intervaloX*intervaloX)));

	cudaMalloc(&d_m, (divX + 2) * (divY + 2) * sizeof(double));

	h_m = (double *) malloc((divX + 2) * (divY + 2) * sizeof(double));

	setupM();

	//Usando "cudaMemcpyToSymbol" para copiar as variaveis da CPU para a GPU
	cudaMemcpyToSymbol(d_intervaloX, &intervaloX, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_intervaloY, &intervaloY, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_denominador1, &denominador1, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_denominador2, &denominador2, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_divX, &divX, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_divY, &divY, sizeof(int), 0, cudaMemcpyHostToDevice);

	cudaMemcpy(d_m, h_m, (divX + 2) * (divY + 2) * sizeof(double), cudaMemcpyHostToDevice);

	dim3 num_threads(TAM_BLOCO, TAM_BLOCO);
	dim3 num_blocos(((divX + 2) + num_threads.x -1)/num_threads.x, ((divY + 2) + num_threads.y -1)/num_threads.y);

	//Calcula a quantidade de posiçes que devem ser calculadas por threads de cada cor
	threadsAzuis = (divX * divY)/2; //Par
	threadsVermelhas = threadsAzuis + (divX * divY)%2; //Impar

	int blocosAzuis, blocosVermelhos;

	blocosAzuis = (threadsAzuis + TAM_BLOCO -1)/TAM_BLOCO;
	blocosVermelhos = (threadsVermelhas + TAM_BLOCO -1)/TAM_BLOCO;
	
	for(i = 0; i < laps; i++){
		calculoAzul<<<blocosAzuis, TAM_BLOCO>>>(d_m);
		calculoVermelho<<<blocosVermelhos, TAM_BLOCO>>>(d_m);
	}
	cudaMemcpy(h_m, d_m, (divX + 2) * (divY + 2) * sizeof(double), cudaMemcpyDeviceToHost);

	printM();

	cudaDeviceReset();

	return 0;
}