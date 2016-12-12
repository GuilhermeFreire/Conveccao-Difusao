# Convecção-Difusão
Algoritmo em paralelo capaz de resolver equações de convecção-difusão sobre uma superfície discreta.

## Como executar
Para rodar a implementação sequencial do Gauss-Seidel, basta executar os seguintes comandos:
```bash
gcc gaussSeidel_SRS.c -o Gauss_Seidel
./Gauss_Seidel <divisões em X> <divisões em Y> <número de iterações>
```
Adicionalmente podemos visualizar a distribuição de calor calculada rodando o script em python após o término do programa principal.
```bash
./Gauss_Seidel <divisões em X> <divisões em Y> <número de iterações>
python heatmap.py
```
Similarmente, para rodar as versões paralelas do programa, basta compilá-las e rodá-las:
```bash
nvcc cuda_gaussSeidel_Mem_Const_SRS.cu -o Gauss_Seidel_CUDA
./Gauss_Seidel_CUDA <divisões em X> <divisões em Y> <número de iterações>
```
A visualização se dá da mesmsa forma que a versão sequencial.
## Visualização ao vivo
Para visualizar a matriz conforme sua evoluçãto nas iterações, vá até a pasta onde o programa principal e o arquivo `live_heatmap.py` estão. Certifique-se de que estão na mesma pasta (talvez seja necessário fazer uma cópia do script em python para a localização do programa principal).
Rode o script `live_heatmap.py`. Se algum erro ocorrer, provavelmente é porque o arquivo `sample.txt` está faltando. Certifique-se que ele existe e está preenchido com alguma matriz válida (isso pode ser feito rodando o programa principal por uma iteração antes de rodar o script em python).
```bash
python live_heatmap.py
```
Com o python rodando, abra outra janela do terminal na mesma pasta e rode a aplicação principal.
```bash
./Gauss_Seidel <divisões em X> <divisões em Y> <número de iterações>
```
A janela do python começará a mostrar a atualização da matriz ao vivo de acordo com o que o programa principal está calculando. Se quiser rodar múltiplas vezes, pode deixar o python aberto e simplesmente rodar o programa principal com outros parâmetros. Trocar as dimensões das matrizes entre execuções com o python aberto pode causar as legendas do gráfico ficarem inconsistentes.
![Malha de 1002x1002 com 100000 iterações](https://github.com/GuilhermeFreire/Conveccao-Difusao/blob/master/images/final.png "Malha de 1002x1002 com 100000 iterações")