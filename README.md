# GUARDIAN v0.0.1 (hackathon-radix)

O GUARDIAN é um sistema de detecção e monitoramento de fadiga com foco em trabalhadores no ambiente offshore.

É composto de duas partes:

- Algoritmo baseado em visão computacional com foco na detecção de fadiga nos olhos de um ser humano. Ele utiliza pequenas amostras do rosto de uma pessoa e através de uma análise do intervalo de tempo entre o fechamento e abertura das pálpebras, determina se a pessoa se encontra em estado de fadiga ou não.

- Dashboard com a relação de todos os colaboradores (foto, nome e ID) e o estado de fadiga de cada um.

## Algoritmo para Detecção de Fadiga

O algoritmo é baseado na biblioteca aberta OpenCV, especificamente a biblioteca MediaPipe que para a identificação do rosto do ser humano mostrou ser uma das mais precisas. Uma webcam é necessária para o funcionamento correto do algoritmo (pode ser a interna em caso de laptop).
 
Para executar basta baixar o arquivo aplication.py e antes de executar é necessário setar a porta correta da webcam. Na linha 35 o índice da função 'VideoCapture(i)' o usuário utilizar inicialmente o valor '0' e ir incrementando em uma unidade até que sua imagem apareça durante a execução.

Durante a execução o usuário poderá observar seu rosto com malhas ao redor de todo o seu rosto. Ao piscar, o algoritmo irá printar o aviso de 'Blink' na tela caso seja uma piscada normal e caso leve mais do que 0.5 segundos, o algoritmo irá exibir um aviso de 'Fadiga'.

![](https://github.com/VHCarvalho/hackaton-radix/misc/DetecçãoFadiga.gif)

## Guardian Dashboard

O Guardian dashboard no momento é um site estático para ilustrar a relação dos colaboradores e o estado de fadiga. O site foi construído em HTML e CSS, sendo o HTML utilizado para fazer o cards dos colaboradores, texto e faixas e o CSS para posicionar todos os elementos principalmente os cards de forma alinhada e dinâmica para que o site não fique comprometido independente de quantidade de cards (colaboradores) tenham.

Para executar basta ter um navegador baixado em seu computador e clicar no arquivo site/Dash_board/index.html.

