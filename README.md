# Convecção-Difusão
Algoritmo em paralelo capaz de resolver equações de convecção-difusão sobre uma superfície discreta.

## Como executar
Para rodar a implementação sequencial do Gauss-Jacobi, basta executar os seguintes comandos:
```bash
gcc gauss_jacobi.c -o Gauss_Jacobi
./Gauss_Jacobi <divisões em X> <divisões em Y>
```
Adicionalmente podemos visualizar a distribuição de calor calculada salvando o resultado da execução em um arquivo `sample.txt` e rodando o script em python.
```bash
./Gauss_Jacobi <divisões em X> <divisões em Y> > sample.txt
python heatmap.py
```
![Malha de 102x102 com 5000 iterações](https://github.com/GuilhermeFreire/Conveccao-Difusao/blob/master/images/100x100_5000.png "Malha de 102x102 com 5000 iterações")