#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define PRECISION 0.00001
#define uN 5.0
#define uS 5.0
#define uW 0.0
#define uE 10.0

double intervaloX, intervaloY;
double denominador1, denominador2;
double* m, *past_m;
int divX, divY;
FILE *output_file;

double a(int i, int j){
	double x = i*intervaloX;
	double y = j*intervaloY;
	return 500 * x * (1 - x) * (0.5 - y);
}

double b(int i, int j){
	double x = i*intervaloX;
	double y = j*intervaloY;
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
	return m[(i)*(divX + 2) + (j)];
}

double u(int i, int j){
	// printf("u(%d, %d) = %lf * %lf + %lf * %lf + %lf * %lf + %lf * %lf\n", i, j, w(i,j), malha(i, j-1), e(i,j), malha(i, j+1), s(i,j), malha(i-1, j), n(i,j), malha(i+1, j));
	// printf("%f * %f + %f * %f + %f * %f + %f * %f\n",w(i,j), malha(i, j-1), e(i,j), malha(i, j+1), s(i,j), malha(i-1, j), n(i,j), malha(i+1, j));
	return w(i,j)*malha(i, j-1) + e(i,j)*malha(i, j+1) + s(i,j)*malha(i-1, j) + n(i,j)*malha(i+1, j);
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

void fprintMat(){
	int i, j;
	for(i = 0; i < divX + 2; i++){
		for(j = 0; j < divY + 2; j++){
			fprintf(output_file, "%lf", m[i*(divX + 2) + j]);
			if(j != divY + 1) fprintf(output_file, " ");
		}
		if(i != divX + 1)
			fprintf(output_file, "\n");
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

int compare(){
	int errors = 0;
	for(int i = 0; i < (divX + 2) * (divY + 2); i++){
		if(fabs(m[i] - past_m[i]) > PRECISION){
			errors++;
		}
	}
	return errors;
}

int main(int argc, char** argv){

	int laps = 0;
	int i, j;
	int diff = 0;

	if(argc < 2){
		printf("Número incorreto de parâmetros:\n");
		printf("Número de divisões faltando:\n");
		printf("\tPara valores iguais: %s <número de divisões>\n", argv[0]);
		printf("\tPara valores diferentes: %s <divisões em X> <divisões em Y>\n", argv[0]);
		exit(-1);
	}
	
	output_file = fopen("test.txt", "w");
	if (output_file == NULL)
	{
	    printf("Error opening file!\n");
	    exit(1);
	}

	divX = atoi(argv[1]);
	divY = (argc > 2)? atoi(argv[2]): divX;

	intervaloX = 1.0/(divX + 1);
	intervaloY = 1.0/(divY + 1);

	denominador1 = 4*(1 + ((intervaloX*intervaloX)/(intervaloY*intervaloY)));
	denominador2 = 4*(1 + ((intervaloY*intervaloY)/(intervaloX*intervaloX)));

	m = (double *) malloc((divX + 2) * (divY + 2) * sizeof(double));
	past_m = (double *) malloc((divX + 2) * (divY + 2) * sizeof(double));

	setupM();

	for(laps = 0; laps < 10000; laps++){
		for(i = 1; i < divX + 1; i++){
			for(j = 1; j < divY + 1; j++){
				m[(i)*(divX + 2) + j] = u(i, j);
			}
		}
		memcpy(past_m, m, (divX + 2) * (divY + 2) * sizeof(double));
		fprintMat();
		fflush(output_file);
		rewind(output_file);
	}

	fclose(output_file);

	return 0;
}