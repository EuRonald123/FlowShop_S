# Implementacoes para o Dataset Taillard

Este documento resume todas as modificacoes e adicoes realizadas no projeto para suportar, executar e avaliar o dataset oficial de Taillard.

## 1. Download e Preparacao (setup_taillard.py)
Foi criado um script dedicado na pasta utils para baixar as 120 instancias padrao de Taillard. Durante o processo de download, o script extrai automaticamente a Best Known Solution (BKS) localizada na segunda linha de cada arquivo de texto. Esses limites sao salvos localmente no arquivo `taillard_bks.json` para agilizar a validacao futura.

## 2. Leitura de Dados (src/flowshop_engine.py)
A classe principal recebeu o metodo estatico `carregar_instancia_taillard()`. Esse metodo interpreta o formato especifico da matriz do Taillard, garantindo a leitura correta das linhas por maquina e ignorando os textos de cabecalho. Tambem inicializa os prazos (due dates) com zero, uma vez que o escopo de Taillard foca primariamente na metrica de Makespan.

## 3. Orquestrador e Catalogo (experimentos_brkga.ipynb)
No notebook principal, a antiga rotina de catalogacao foi substituida pela funcao `construir_catalogo_taillard()`. Ela processa todos os arquivos `ta*.txt`, extrai de forma dinamica o numero de maquinas e tarefas, e anexa o BKS diretamente ao DataFrame. A variavel de "grupo" presente nas versoes antigas das instancias foi fixada, mantendo a compatibilidade do modelo de CSV.

## 4. Metrica de Comparacao (RPD)
O parametro de avaliacao "GAP", calculado contra o limite inferior teorico de Potts, foi removido. Em seu lugar, foi implementada a metrica oficial da literatura: RPD (Relative Percentage Deviation). A formula exata adicionada ao loop principal e:

RPD = ((Makespan_Obtido - BKS) / BKS) * 100.0

A estrutura do arquivo CSV gerado (atraves da lista `CAMPOS_CSV`) e o codigo de agregacao do Pandas foram reescritos para utilizar as colunas `bks` e `rpd_medio`.

## 5. Criterio de Parada e Historico (brkga_scs_solver.py)
A condicao de parada do algoritmo agora segue o metodo SOTA de limite de tempo proporcional. O laco e interrompido dinamicamente com base na dimensao da instancia pela formula `(n * m * t_param / 2) / 1000`. O historico de convergencia devolvido pelo solver foi modificado para salvar e desempacotar tres valores independentes: CFOs consumidos, tempo real decorrido e makespan atual. A funcao de extracao `amostrar_historico` foi blindada para processar esse novo formato no orquestrador.
