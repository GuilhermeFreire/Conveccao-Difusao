#include <stdio.h>
#include <stdlib.h>

#define PRECISION 0.00001
#define uN 5.0
#define uS 5.0
#define uW 0.0
#define uE 10.0

double intervaloX, intervaloY;
double denominador1, denominador2;
double* m;
int divX, divY;

double a(int i, int j){
	double x = i*intervaloX;
	double y = i*intervaloY;
	return 500 * x * (1 - x) * (0.5 - y);
}

double b(int i, int j){
	double x = i*intervaloX;
	double y = i*intervaloY;
	return 500 * y * (y - 1) * (x - 0.5);
}

double n(int i, int j){
	return (2.0 - intervaloX * b(i,j))/denominador2;
}
double s(int i, int j){
	return (2.0 + intervaloX * b(i,j))/denominador2;
}
double e(int i, int j){
	return (2.0 - intervaloX * a(i,j))/denominador1;
}
double w(int i, int j){
	return (2.0 + intervaloX * a(i,j))/denominador1;
}

double malha(int i, int j){
	// //OBS: Casos de canto não importam, pois nunca serão usados no problema.
	// //e.g. u(0,0) ou u(1,0) nunca serão usados no problema então tanto faz
	// //o valor retornado.
	// if(i == 0){
	// 	//uW foi escolhido para representar o uO do trabalho. 
	// 	//W vem de 'West', visto que todas as variáveis exceto essa foram
	// 	//nomeadas de acordo com sua cardinalidade em inglês
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
	return m[(i)*(divX + 2) + (j)];
}

double u(int i, int j){
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
	return w(i,j)*malha(i, j-1) + e(i,j)*malha(i, j+1) + s(i,j)*malha(i-1, j) + n(i,j)*malha(i+1, j);
}

void printM(){
	int i = 0, j = 0;
	for(i = 0; i < divX + 2; i++){
		printf("%lf ", malha(0,i));
	}
	printf("\n");

	for(i = 0; i < divX; i++){
		for(j = 0; j < divY; j++){
			if(j == 0){
				printf("%lf ", malha(i+1,0));
			}

			printf("%lf ", m[i* divX + j]);
			
			if(j == divY - 1){
				printf("%lf ", malha(i+1,divY + 1));
			}
		}
		printf("\n");
	}

	for(i = 0; i < divX + 2; i++){
		printf("%lf ", malha(divX + 1,i));
	}
	printf("\n");
}

void printMat(){
	int i, j;
	for(i = 0; i < divX + 2; i++){
		for(j = 0; j < divY + 2; j++){
			printf("%lf", m[i*(divX + 2) + j]);
			if(j != divY + 1) printf(" ");
		}
		if(i != divX + 1)
			printf("\n");
	}
}

void setupM(){
	int i;
	for(i = 0; i < divY + 2; i++){
		m[i * (divX + 2)] = uW;
		m[i*(divX + 2) + divY + 1] = uE;
	}
	for(i = 0; i < divX + 2; i++){
		m[i] = uN;
		m[(divX + 1)*(divX + 2) + i] = uS;
	}
}

int main(int argc, char** argv){

	int laps = 0;
	int i, j;

	if(argc < 2){
		printf("Número incorreto de parâmetros:\n");
		printf("Número de divisões faltando:\n");
		printf("\tPara valores iguais: %s <número de divisões>\n", argv[0]);
		printf("\tPara valores diferentes: %s <divisões em X> <divisões em Y>\n", argv[0]);
		exit(-1);
	}

	divX = atoi(argv[1]);
	divY = (argc > 2)? atoi(argv[2]): divX;

	intervaloX = 1.0/(divX + 1);
	intervaloY = 1.0/(divY + 1);

	denominador1 = 4*(1 + ((intervaloX*intervaloX)/(intervaloY*intervaloY)));
	denominador2 = 4*(1 + ((intervaloY*intervaloY)/(intervaloX*intervaloX)));

	m = (double *) malloc((divX + 2) * (divY + 2) * sizeof(double));

	setupM();

	for(laps = 0; laps < 10000; laps++){
		for(i = 1; i < divX + 1; i++){
			for(j = 1; j < divY + 1; j++){
				// printf("u(%d, %d) = %lf,\n", i, j, u(i, j));
				m[(i)*(divX + 2) + j] = u(i, j);
				// printMat();
			}
			// printf("\n");
		}
		// printM();
		// printf("\n");
	}

	printMat();

	return 0;
}