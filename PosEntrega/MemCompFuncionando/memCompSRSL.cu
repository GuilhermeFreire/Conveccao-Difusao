#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PRECISION 0.00001
#define TAM_BLOCO 8
#define uN 5.0
#define uS 5.0
#define uW 0.0
#define uE 10.0

//Variáveis CPU
double h_h1, h_h2;
double h_denominador1, h_denominador2;
double *h_m, *d_m;
double h_parcial1, h_parcial2;
double h_cosH1PI, h_cosH2PI;
int h_dimensaoX, h_dimensaoY, laps = 0, i;

//Variáveis GPU
__constant__ double d_h1, d_h2;
__constant__ double d_denominador1, d_denominador2;
__constant__ int d_dimensaoX, d_dimensaoY;
__constant__ double d_parcial1, d_parcial2;
__constant__ double d_cosH1PI, d_cosH2PI;

__device__ __shared__ double subMatriz[TAM_BLOCO][TAM_BLOCO];

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
	return 500 * y * (1 - y) * (x - 0.5);
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


__device__ double pontosExternos(int i, int j, int x, int y, double *m){

	double temp = 0;

	temp += w(x,y) * m[(x - 1) * d_dimensaoY + y];
	temp += e(x,y) * m[(x + 1) * d_dimensaoY + y];
	temp += s(x,y) * m[x * d_dimensaoY + (y - 1)];
	temp += n(x,y) * m[x * d_dimensaoY + (y + 1)];

	return temp;

}

__device__ double pontosInternos(int i, int j, int x, int y, double *m){

	double temp = 0;

	temp += w(x,y) * subMatriz[i - 1][j];
	temp += e(x,y) * subMatriz[i + 1][j];
	temp += s(x,y) * subMatriz[i][j - 1];
	temp += n(x,y) * subMatriz[i][j + 1];

	return temp;

}

__device__ double letraGrega(int i, int j){

	double raiz1, raiz2, total;

	raiz1 = e(i,j) * w(i,j);
	raiz1 = pow(raiz1, 0.5);
	raiz1 = raiz1 * cos(d_h1*M_PI);

	raiz2 = s(i,j) * n(i,j);
	raiz2 = pow(raiz2, 0.5);
	raiz2 = raiz2 * cos(d_h2*M_PI);

	total = 2*(raiz1 + raiz2);
	return total;
}

__device__ double funcaoOmega(int i, int j){

	double raiz, total;

	raiz = 1 - pow(letraGrega(i, j), 2);
	raiz = pow(raiz, 0.5);

	total = 2/(1 + raiz);

	return total;
}


//Kernels principais do programa. Cada um trabalho em um conjunto de pontos da matriz
//fazendo uma media ponderada entre o valor atual do ponto que está sendo analisado e 
//seus quatro pontos adjacentes. O quanto cada valor vai pesar é determinado pelo ômega
//da funcao que, nesse caso, é fixo
__global__ void vermelhos(double *m){
	int tidx = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if(tidx > (d_dimensaoX - 2) || tidy > (d_dimensaoY - 2)){
		return;
	}


	//printf("%d %d\n", tidx, tidy);

	if((tidx + tidy) % 2 == 1){
		subMatriz[threadIdx.x][threadIdx.y] = m[tidx * d_dimensaoY + tidy];
	}

	__syncthreads();

	
	if((tidx + tidy) % 2 == 0){
		if(threadIdx.x == 0 || threadIdx.x == (TAM_BLOCO - 1) || threadIdx.y == 0 || threadIdx.y == (TAM_BLOCO - 1)){
			double omega = funcaoOmega(tidx, tidy);
			m[tidx * d_dimensaoY + tidy] *= (1 - omega);
		 	m[tidx * d_dimensaoY + tidy] += omega * pontosExternos(threadIdx.x, threadIdx.y, tidx, tidy, m);
		}else{
			double omega = funcaoOmega(tidx, tidy);
			m[tidx * d_dimensaoY + tidy] *= (1 - omega);
		 	m[tidx * d_dimensaoY + tidy] += omega * pontosInternos(threadIdx.x, threadIdx.y, tidx, tidy, m);
		}
	}
}

__global__ void azuis(double *m){
	int tidx = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if(tidx > (d_dimensaoX - 2) || tidy > (d_dimensaoY - 2)){
		return;
	}

	if((tidx + tidy) % 2 == 0){
		subMatriz[threadIdx.x][threadIdx.y] = m[tidx * d_dimensaoY + tidy];
	}

	__syncthreads();

	
	if((tidx + tidy) % 2 == 1){
		if(threadIdx.x == 0 || threadIdx.x == (TAM_BLOCO - 1) || threadIdx.y == 0 || threadIdx.y == (TAM_BLOCO - 1)){
			double omega = funcaoOmega(tidx, tidy);
			m[tidx * d_dimensaoY + tidy] *= (1 - omega);
		 	m[tidx * d_dimensaoY + tidy] += omega * pontosExternos(threadIdx.x, threadIdx.y, tidx, tidy, m);
		}else{
			double omega = funcaoOmega(tidx, tidy);
			m[tidx * d_dimensaoY + tidy] *= (1 - omega);
		 	m[tidx * d_dimensaoY + tidy] += omega * pontosInternos(threadIdx.x, threadIdx.y, tidx, tidy, m);
		}
	}
}

int main(int argc, char** argv){

	//Especificacoes iniciais para garantir que o programa será rodado com as 
	//condicoes iniciais corretas
	if(argc != 4){
		printf("Número incorreto de parâmetros:\n");
		printf("Insira as dimensoes e a quantidade de iterações\n");
		printf("\tUtilize o formato: %s <Dimensao X> <Dimensao Y> <Iterações>\n", argv[0]);
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

	h_cosH1PI = cos(h_h1*M_PI);
	h_cosH2PI = cos(h_h2*M_PI);

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
	cudaMemcpyToSymbol(d_cosH1PI, &h_cosH1PI, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_cosH2PI, &h_cosH2PI, sizeof(double), 0, cudaMemcpyHostToDevice);

	//Iniciando a contagem do tempo
	start = clock();

	//Calculando a quantidade de blocos e threads que serao lancados
	dim3 nthreads(TAM_BLOCO,TAM_BLOCO);
	dim3 nblocos(((h_dimensaoX - 2) + nthreads.x - 1)/nthreads.x, ((h_dimensaoY - 2) + nthreads.y - 1)/nthreads.y);

	//Fazendo os cálculos
	for(i = 0; i < laps; i++){
		vermelhos<<<nblocos, nthreads>>>(d_m);
		azuis<<<nblocos, nthreads>>>(d_m);
	}

	//Trazendo a matriz de volta para a CPU
	cudaMemcpy(h_m, d_m, h_dimensaoX * h_dimensaoY * sizeof(double), cudaMemcpyDeviceToHost);

	//Reseta a GPU para liberar todos os recursos
	cudaDeviceReset();

	//Imprimindo a matriz no arquivo e fechando-o
	arquivo = fopen("sample.txt", "w");
	printMat();
	fclose(arquivo);

	//Termina de calcular o tempo que demorou o programa
	end = clock();
	tempo = ((double)  (end - start))/CLOCKS_PER_SEC;
	printf("%lf;", tempo);

	return 0;
}