#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PRECISION 0.00001
#define TAM_BLOCO 16
#define uN 5.0
#define uS 5.0
#define uW 0.0
#define uE 10.0

//Variáveis CPU
double h_h1, h_h2;
double h_denominador1, h_denominador2;
double *h_m, *d_m;
int h_dimensaoX, h_dimensaoY, laps = 0, i;

//Variáveis GPU
__device__ double omega = 1.5;
__device__ double d_h1, d_h2;
__device__ double d_denominador1, d_denominador2;
__device__ int d_dimensaoX, d_dimensaoY;
__device__ double d_PI = 3.14159265358979323846;

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
	double y = i * d_h2;
	return 500 * x * (1 - x) * (0.5 - y);
}

__device__ double b(int i, int j){
	double x = i * d_h1;
	double y = i * d_h2;
	return 500 * y * (y - 1) * (x - 0.5);
}


//Funcoes "n", "s", "w", "e" especificadas pelo problema
__device__ double n(int i, int j){
	return (2.0 - d_h2 * b(i,j))/d_denominador2;
}
__device__ double s(int i, int j){
	return (2.0 + d_h2 * b(i,j))/d_denominador2;
}
__device__ double e(int i, int j){
	return (2.0 - d_h1 * a(i,j))/d_denominador1;
}
__device__ double w(int i, int j){
	return (2.0 + d_h1 * a(i,j))/d_denominador1;
}


//Funcao que faz a media ponderada dos valores vizinhos ao ponto que está sendo atualizado
__device__ double somaDosPontosVizinhos(int i, int j, double *m){

	double temp = 0;

	temp += w(i,j) * m[(i - 1) * d_dimensaoY + j];
	temp += e(i,j) * m[(i + 1) * d_dimensaoY + j];
	temp += s(i,j) * m[i * d_dimensaoY + (j - 1)];
	temp += n(i,j) * m[i * d_dimensaoY + (j + 1)];

	return temp;
}

//As funcoes "funcaoOmega()" e " letraGrega()" abaixo fazem um calculo do omega especifico
//para cada ponto que etá sendo analisado. A ideia é a mesma do outro programa,a de fazer uma
//media ponderada, porem aqui as condicoes sao mais especificas
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
//da funcao que, nesse caso, é calculado mais a fundo
__global__ void vermelhos(double *m){
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	if(tidx != 0 && tidx < d_dimensaoX - 1 && tidy != 0 && tidy < d_dimensaoY - 1){
		if((tidx + tidy) % 2 == 0){
			double omega = funcaoOmega(tidx, tidy);
			m[tidx * d_dimensaoY + tidy] *= (1 - omega);
			m[tidx * d_dimensaoY + tidy] += omega * somaDosPontosVizinhos(tidx, tidy, m);
		}
	}
}

__global__ void azuis(double *m){
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	if(tidx != 0 && tidx < d_dimensaoX - 1 && tidy != 0 && tidy < d_dimensaoY - 1){
		if((tidx + tidy) % 2 == 1){
			double omega = funcaoOmega(tidx, tidy);
			m[tidx * d_dimensaoY + tidy] *= (1 - omega);
			m[tidx * d_dimensaoY + tidy] += omega * somaDosPontosVizinhos(tidx, tidy, m);
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

	//Iniciando a contagem do tempo
	start = clock();

	//Calculando a quantidade de blocos e threads que serao lancados
	dim3 nthreads(TAM_BLOCO,TAM_BLOCO);
	dim3 nblocos((h_dimensaoX + nthreads.x - 1)/nthreads.x, (h_dimensaoY + nthreads.y - 1)/nthreads.y);

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
	printf("Tempo total: %lfs...\n", tempo);

	return 0;
}