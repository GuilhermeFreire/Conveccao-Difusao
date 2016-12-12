#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define PRECISION 0.00001
#define TAM_BLOCO 32
#define uN 5.0
#define uS 5.0
#define uW 0.0
#define uE 10.0

//Variáveis GPU
__constant__ double omega = 1.5;
__constant__ double d_h1, d_h2;
__constant__ double d_denominador1, d_denominador2;
__constant__ int d_dimensaoX, d_dimensaoY;
__constant__ double d_parcial1, d_parcial2;

//Variáveis CPU
double h_h1, h_h2;
double h_denominador1, h_denominador2;
double *h_m, *d_m;
double h_parcial1, h_parcial2;
int h_dimensaoX, h_dimensaoY, laps = 0, i;

__device__ __shared__ double shared_matrix[TAM_BLOCO*TAM_BLOCO];


FILE *arquivo;

 clock_t start, end;
 double tempo;

//Funções da CPU

//Funcao que imprime a matriz no arquivo de saida
void printMat(){
	int i, j;
	for(i = 0; i < h_dimensaoX; i++){
		for(j = 0; j < h_dimensaoY; j++){
			fprintf(arquivo, "%lf", h_m[i * h_dimensaoY + j]);
			if(j != h_dimensaoY - 1) fprintf(arquivo, " ");
		}
		if(i != h_dimensaoX - 1)
			fprintf(arquivo, "\n");
	}
}

//Funcao que inicializa a matriz com os valores de contorno especificados pelo problema
void setupM(){
	int i,j;
	for(i = 0; i < h_dimensaoX; i++){
		for(j = 0; j < h_dimensaoY; j++){
			if(i == 0){
				h_m[i * h_dimensaoY + j] = uN;
			}else if(i == (h_dimensaoX - 1)){
				h_m[i * h_dimensaoY + j] = uS;
			}else if(j == 0){
				h_m[i * h_dimensaoY + j] = uW;
			}else if(j == h_dimensaoY - 1){
				h_m[i * h_dimensaoY + j] = uE;
			}
		}
	}
}

//Funções da GPU

//Funcoes "a" e "b" especificada pelo problema
__device__ double a(int i, int j){
	double x = i * d_h1;
	double y = j * d_h2;
	return 500 * x * (1 - x) * (0.5 - y);
}

__device__ double b(int i, int j){
	double x = i * d_h1;
	double y = j * d_h2;
	return 500 * y * (y - 1) * (x - 0.5);
}


//Funcoes "n", "s", "w", "e" especificadas pelo problema
__device__ double n(int i, int j){
	return (d_parcial2 - (d_h2 * b(i,j))/d_denominador2);
}
__device__ double s(int i, int j){
	return (d_parcial2 + (d_h2 * b(i,j))/d_denominador2);
}
__device__ double e(int i, int j){
	return (d_parcial1 - (d_h1 * a(i,j))/d_denominador1);
}
__device__ double w(int i, int j){
	return (d_parcial1 + (d_h1 * a(i,j))/d_denominador1);
}


__device__ double pontosInternos(int i_local, int j_local, int i, int j, double *m){

	double temp = 0;

	temp += w(i,j) * shared_matrix[(i_local - 1) * TAM_BLOCO + j_local];
	temp += e(i,j) * shared_matrix[(i_local + 1) * TAM_BLOCO + j_local];
	temp += s(i,j) * shared_matrix[i_local * TAM_BLOCO + (j_local - 1)];
	temp += n(i,j) * shared_matrix[i_local * TAM_BLOCO + (j_local + 1)];

	return temp;
}

__device__ double pontosExternos(int i, int j, double *m){

	double temp = 0;

	temp += w(i,j) * m[(i - 1) * d_dimensaoY + j];
	temp += e(i,j) * m[(i + 1) * d_dimensaoY + j];
	temp += s(i,j) * m[i * d_dimensaoY + (j - 1)];
	temp += n(i,j) * m[i * d_dimensaoY + (j + 1)];

	return temp;
}
//Kernels principais do programa. Cada um trabalho em um conjunto de pontos da matriz
//fazendo uma media ponderada entre o valor atual do ponto que está sendo analisado e 
//seus quatro pontos adjacentes. O quanto cada valor vai pesar é determinado pelo ômega
//da funcao que, nesse caso, é fixo
__global__ void vermelhos(double *m){
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	// if(tidx =7 && tidy == 85){
	// 	printf("%d %d\n", threadIdx.x, threadIdx.y);
	// }

	int i_bloco = threadIdx.x;
	int j_bloco = threadIdx.y;

	//Restringindo as threads ao tamanho da matriz
	if(tidx > 0 && tidy > 0 && tidx < d_dimensaoX - 1 && tidy < d_dimensaoY - 1){


			//Se for azul traz seu valor pra memória compartilhada
			if((i_bloco + j_bloco)%2 == 1){
				shared_matrix[i_bloco * TAM_BLOCO + j_bloco] = m[tidx * d_dimensaoY + tidy];
			}
			__syncthreads();

			//Vê se é um ponto externo ou interno e calcula seu valor de acordo
			if((i_bloco + j_bloco)%2 == 0){
				if(threadIdx.x > 0 && threadIdx.x < TAM_BLOCO - 2 && threadIdx.y > 0 && threadIdx.y < TAM_BLOCO - 2){
					m[tidx * d_dimensaoY + tidy] *= (1 - omega);
			 		m[tidx * d_dimensaoY + tidy] += omega * pontosInternos(i_bloco, j_bloco, tidx, tidy, m);
				}else{
					m[tidx * d_dimensaoY + tidy] *= (1 - omega);
			 		m[tidx * d_dimensaoY + tidy] += omega * pontosExternos(tidx, tidy, m);
				}
			}
	}
}

__global__ void azuis(double *m){
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	int i_bloco = threadIdx.x;
	int j_bloco = threadIdx.y;

	if(tidx > 0 && tidy > 0 && tidx < d_dimensaoX - 1 && tidy < d_dimensaoY - 1){


		if(tidx < d_dimensaoX && tidy < d_dimensaoY){

				shared_matrix[i_bloco * TAM_BLOCO + j_bloco] = m[tidx * d_dimensaoY + tidy];
			}
		

			__syncthreads();

			if((i_bloco + j_bloco)%2 == 1){
				if(threadIdx.x > 0 && threadIdx.x < TAM_BLOCO - 2 && threadIdx.y > 0 && threadIdx.y < TAM_BLOCO - 2){
					m[tidx * d_dimensaoY + tidy] *= (1 - omega);
			 		m[tidx * d_dimensaoY + tidy] += omega * pontosInternos(i_bloco, j_bloco, tidx, tidy, m);
				}else{
					m[tidx * d_dimensaoY + tidy] *= (1 - omega);
			 		m[tidx * d_dimensaoY + tidy] += omega * pontosExternos(tidx, tidy, m);
				}
			}
	}
}

int main(int argc, char** argv){

	cudaDeviceReset();

	//Especificacoes iniciais para garantir que o programa será rodado com as 
	//condicoes iniciais corretas
	if(argc != 4){
		printf("Número incorreto de parâmetros:\n");
		printf("Insira as dimensoes e a quantidade de iterações\n");
 		exit(-1);
	}

	//Inicializando todos os valores necessários para transferir para a GPU e para realizar 
	//os calculos do programa
	h_dimensaoX = atoi(argv[1]);
	h_dimensaoY = atoi(argv[2]);
	laps = atoi(argv[3]); 

	h_h1 = 1.0/(h_dimensaoX + 1);
	h_h2 = 1.0/(h_dimensaoY + 1);

	h_dimensaoX += 2;
	h_dimensaoY += 2;

	h_denominador1 = 4*(1 + (pow(h_h1,2)/pow(h_h2,2)));
	h_denominador2 = 4*(1 + (pow(h_h2,2)/pow(h_h1,2)));

	h_parcial1 = 2/h_denominador1;
	h_parcial2 = 2/h_denominador2;

	//Alocando a matriz na CPU e inicializando
	h_m = (double *) calloc(h_dimensaoX * h_dimensaoY, sizeof(double));
	setupM();

	//Alocando a matriz na GPU
	cudaMalloc(&d_m, h_dimensaoX * h_dimensaoY * sizeof(double));

	//Transferindo as informações necessárias para a GPU
	cudaMemcpy(d_m, h_m, h_dimensaoX * h_dimensaoY * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_denominador1, &h_denominador1, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_denominador2, &h_denominador2, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_dimensaoX, &h_dimensaoX, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_dimensaoY, &h_dimensaoY, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_h1, &h_h1, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_h2, &h_h2, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_parcial1, &h_parcial1, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_parcial2, &h_parcial2, sizeof(double), 0, cudaMemcpyHostToDevice);

	//Iniciando a contagem do tempo
	start = clock();

	//Calculando a quantidade de blocos e threads que serao lancados
	dim3 nthreads(TAM_BLOCO,TAM_BLOCO);
	dim3 nblocos((h_dimensaoX + nthreads.x - 1)/nthreads.x, (h_dimensaoY + nthreads.y - 1)/nthreads.y);

	printf("%d %d\n", nblocos.x , nblocos.y);
	// int j;
	// for(i = 0; i < h_dimensaoX; i++){
	// 	for(j = 0; j < h_dimensaoY; j++){
	// 		printf("%lf ", h_m[i *h_dimensaoY +j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("\n");

	//Fazendo os cálculos
	for(i = 0; i < laps; i++){

		vermelhos<<<nblocos, nthreads>>>(d_m);
		//cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );
		// if(laps == 50){
		 	
		// }
		azuis<<<nblocos, nthreads>>>(d_m);
		//cudaDeviceSynchronize();
		//printf("oi %d\n",i);
		//gpuErrchk( cudaPeekAtLastError() );

	}

	//Trazendo a matriz de volta para a CPU
	cudaMemcpy(h_m, d_m, h_dimensaoX * h_dimensaoY * sizeof(double), cudaMemcpyDeviceToHost);

	//Reseta a GPU para liberar todos os recursos
	cudaDeviceReset();

	//Imprimindo a matriz no arquivo e fechando-o
	arquivo = fopen("sample.txt", "w+");
	printMat();
	fclose(arquivo);

	//Termina de calcular o tempo que demorou o programa
	end = clock();
	tempo = ((double)  (end - start))/CLOCKS_PER_SEC;
	printf("Tempo total: %lfs...\n", tempo);

	return 0;
}
