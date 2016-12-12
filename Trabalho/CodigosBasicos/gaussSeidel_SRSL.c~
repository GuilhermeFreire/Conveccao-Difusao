#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PRECISION 0.00001
#define uN 5.0
#define uS 5.0
#define uW 0.0
#define uE 10.0
#define M_PI 3.14159265358979323846

double h1, h2;
double denominador1, denominador2;
double* m;
int dimensaoX, dimensaoY;

FILE *arquivo;

 clock_t start, end;
 double tempo;

double a(int i, int j){
	double x = i*h1;
	double y = j*h2;
	return 500 * x * (1 - x) * (0.5 - y);
}

double b(int i, int j){
	double x = i*h1;
	double y = j*h2;
	return 500 * y * (1 - y) * (x - 0.5);
}

double n(int i, int j){
	return (2.0 - h2 * b(i,j))/denominador2;
}
double s(int i, int j){
	return (2.0 + h2 * b(i,j))/denominador2;
}
double e(int i, int j){
	return (2.0 - h1 * a(i,j))/denominador1;
}
double w(int i, int j){
	return (2.0 + h1 * a(i,j))/denominador1;
}

void printMat(){
	int i, j;
	for(i = 0; i < dimensaoX; i++){
		for(j = 0; j < dimensaoY; j++){
			fprintf(arquivo, "%lf", m[i*dimensaoY + j]);
			if(j != dimensaoY - 1) fprintf(arquivo, " ");
		}
		if(i != dimensaoX - 1)
			fprintf(arquivo, "\n");
	}
	//printf("\n");
}

void setupM(){
	int i,j;
	for(i = 0; i < dimensaoX; i++){
		for(j = 0; j < dimensaoY; j++){
			//printf("1 %d  %d\n", i, j);
			if(i == 0){
				m[i * dimensaoY + j] = uN;
			}else if(i == (dimensaoX - 1)){
				m[i * dimensaoY + j] = uS;
			}else if(j == 0){
				m[i * dimensaoY + j] = uW;
			}else if(j == dimensaoY - 1){
				m[i * dimensaoY + j] = uE;
			}
		}
	}
}

double letraBizarra(int i, int j){

	double raiz1, raiz2, total;

	raiz1 = e(i,j) * w(i,j);
	raiz1 = pow(raiz1, 0.5);
	raiz1 = raiz1 * cos(h1*M_PI);

	raiz2 = s(i,j) * n(i,j);
	raiz2 = pow(raiz2, 0.5);
	raiz2 = raiz2 * cos(h2*M_PI);

	total = 2*(raiz1 + raiz2);
	return total;
}

double funcaoOmega(int i, int j){

	double raiz, total;

	raiz = 1 - pow(letraBizarra(i, j), 2);
	raiz = pow(raiz, 0.5);

	total = 2/(1 + raiz);

	return total;
}

double somaDosPontosVizinhos(int i, int j){

	double temp = 0;

	temp += w(i,j) * m[(i - 1) * dimensaoY + j];
	temp += e(i,j) * m[(i + 1) * dimensaoY + j];
	temp += s(i,j) * m[i * dimensaoY + (j - 1)];
	temp += n(i,j) * m[i * dimensaoY + (j + 1)];

	return temp;
}

int main(int argc, char** argv){

	int laps = 0;
	int i, j, k;

	if(argc != 4){
		printf("Número incorreto de parâmetros:\n");
		printf("Insira as dimensoes e a quantidade de iterações\n");
		printf("\tUtilize o formato: %s <Dimensao X> <Dimensao Y> <Iterações>\n", argv[0]);
		exit(-1);
	}

	dimensaoX = atoi(argv[1]);
	dimensaoY = atoi(argv[2]);
	laps = atoi(argv[3]); 

	h1 = 1.0/(dimensaoX + 1);
	h2 = 1.0/(dimensaoY + 1);

	denominador1 = 4*(1 + (pow(h1,2)/pow(h2,2)));
	denominador2 = 4*(1 + (pow(h2,2)/pow(h1,2)));

	dimensaoX += 2;
	dimensaoY += 2;

	m = (double *) calloc(dimensaoX * dimensaoY, sizeof(double));

	setupM();

	start = clock();


	//Dois For's um depois do outro pra calcular todos os pontos. O primeiro
	//calcula todos os vermelhos e o segundo todos os azuis (diferenciados por
	//se i+j é par ou ímpar). Tentei seguir bem fielmente o que tá escrito no 
	//pdf da descrição do trabalho. Essa versão é a do método de Gauss-Seidel
	//com sobre-relaxação sucessiva local que nem tá descrito no pdf. O global
	//é uma generalização desse pelo que entendi, então deve ser mais fácil 
	//de implementar
	for(k = 0; k < laps; k++){
		for(i = 1; i < (dimensaoX - 1); i++){
			for(j = 1; j < (dimensaoY - 1); j++){
				if((i + j) % 2 == 0){
					double omega = funcaoOmega(i,j);
					m[i * dimensaoY + j] *= (1 - omega);
					m[i * dimensaoY + j] += omega * somaDosPontosVizinhos(i,j);
				}
			}
		}

		for(i = 1; i < (dimensaoX - 1); i++){
			for(j = 1; j < (dimensaoY - 1); j++){
				if((i + j) % 2 == 1){
					double omega = funcaoOmega(i,j);
					m[i * dimensaoY + j] *= (1 - omega);
					m[i * dimensaoY + j] += omega * somaDosPontosVizinhos(i,j);
				}
			}
		}
	}

	arquivo = fopen("sample.txt", "w");
	printMat();
	fclose(arquivo);

	end = clock();

	tempo = ((double)  (end - start))/CLOCKS_PER_SEC;
	printf("Tempo total: %lfs...\n", tempo);

	return 0;
}
