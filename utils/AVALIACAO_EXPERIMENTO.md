# Avaliação dos Experimentos e Análise do Projeto Flow Shop

## Veredito Direto sobre suas Conclusões

> [!IMPORTANT]
> **Resumo Executivo:**
> 1. **Sobre o BRKGA-SCS ser muito superior aos outros 3 BRKGA (100 gerações):** O resultado nos dados é real, mas a causa não se deve apenas a uma "melhor sintonia" do BRKGA. O SCS venceu porque opera como um algoritmo em **dois espaços (PFS + NPFS)**, reduzindo a dimensionalidade de $80$ para $16$ variáveis e usando **Cross-Space Knowledge Transfer + Busca Local (Hooke & Jeeves)**, enquanto os outros 3 BRKGA tentavam resolver o espaço NPFS às cegas em 80 dimensões contínuas.
> 2. **Sobre convergirem e serem equivalentes em 500 gerações (SCS puro vs SCS Optuna):** Eles de fato empataram no makespan final (~$951$), mas **a premissa de que "rodaram 500 gerações e convergiram" é uma ilusão causada pelo código**. O algoritmo foi interrompido na geração **~60** por estourar o teto de 50.000 avaliações, e o script de amostragem repetiu o valor estagnado da geração 60 em todas as colunas de `mks_gen80` até `mks_gen500`.

---

## 1. Por que o BRKGA-SCS foi tão superior em 100 gerações?

No arquivo [resultados_brkga_TODOS_100_gerac.csv](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/resultados/resultados_brkga_TODOS_100_gerac.csv), os números observados foram:

| Método | Makespan Médio | Gap Médio (%) | Tempo Médio (s) | Espaço de Busca | Dimensões ($D$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `fixo_default` | 1211.24 | 61.69% | 2.29s | NPFS direto | $16 \times 5 = \mathbf{80}$ |
| `fixo_optuna` | 1195.23 | 59.56% | 2.70s | NPFS direto | $16 \times 5 = \mathbf{80}$ |
| `fuzzy` | 1229.00 | 64.09% | 5.17s | NPFS direto | $16 \times 5 = \mathbf{80}$ |
| **`brkga_scs`** | **961.95** | **28.69%** | **1.06s** | **PFS + Transferência NPFS + BL** | $\mathbf{16}$ |

### A causa matemática e arquitetural:

* **As variantes `fixo_default`, `fixo_optuna` e `fuzzy`** (definidas em [brkga_solver.py:16-31](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/src/solvers/brkga_solver.py#L16-L31)):
  Evoluem cromossomos de tamanho $N \times M$ ($16 \times 5 = 80$ variáveis contínuas). O espaço de busca para permutações independentes nas 5 máquinas é de $(16!)^5 \approx 3.7 \times 10^{66}$. Em apenas 100 gerações sem busca local, um algoritmo genético fica completamente perdido nessa hiperesfera de 80 dimensões.
* **O `brkga_scs`** (implementado em [brkga_scs_solver.py:236-245](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/src/solvers/brkga_scs_solver.py#L236-L245)):
  A população macro evolui no espaço **PFS** ($D = 16$). O espaço é de apenas $16! \approx 2 \times 10^{13}$.
  * **Teste comparativo isolado:** um BRKGA puro operando em PFS atinge makespan **884.0**, enquanto o mesmo BRKGA puro operando em NPFS direto atinge **1412.0**!
  * Quando o SCS detecta um cluster promissor no PFS, projeta a solução para o NPFS via `np.tile(c.centro, m)` ([brkga_scs_solver.py:309](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/src/solvers/brkga_scs_solver.py#L309)) e executa a busca local Hooke & Jeeves ([brkga_scs_solver.py:102-147](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/src/solvers/brkga_scs_solver.py#L102-L147)).
  * **Conclusão:** O SCS venceu porque é um **algoritmo memético híbrido bi-fásico com redução drástica de dimensionalidade**, e não apenas por mérito das probabilidades genéticas.

---

## 2. A verdade sobre as "500 gerações" e o Optuna

Ao rodar o experimento em [resultados_brkga_scs_optuna_500_gerac.csv](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/resultados/resultados_brkga_scs_optuna_500_gerac.csv), os resultados foram:

* `brkga_scs`: Makespan médio **951.41** | Gap médio **27.26%**
* `brkga_scs_optuna`: Makespan médio **951.03** | Gap médio **27.21%**

Ambos parecem estáticos a partir da geração 60/80 até a geração 500:

```text
brkga_scs:        gen20: 953.16 -> gen60: 951.41 -> gen100..500: 951.41 (estagnado)
brkga_scs_optuna: gen20: 954.19 -> gen60: 951.17 -> gen100..500: 951.03 (estagnado)
```

### O que realmente ocorreu nos bastidores:

1. **O algoritmo nunca chegou na geração 500:**
   No arquivo [adapter.py:521-548](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/src/solvers/adapter.py#L521-L548), o orçamento total de avaliações é fixado por:
   $$\text{max\_evals\_total} = n\_gen \times tam\_pop = 500 \times 100 = 50.000\text{ avaliações}$$
   Como a busca local Hooke & Jeeves consome até 500 avaliações de função objetivo a cada ativação de cluster, o teto de 50.000 avaliações foi atingido **por volta da geração 60 a 62**. O loop parou pela condição `if evals_count >= max_evals_total: break`.
2. **O congelamento no CSV:**
   No notebook [brkga_foco_v2.ipynb](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/brkga_foco_v2.ipynb), a função de amostragem faz:
   ```python
   idx = min(gen - 1, len(history) - 1)
   amostras[f'mks_gen{gen}'] = float(history[idx])
   ```
   Como `len(history)` tem apenas 62 registros, para todo `gen >= 80`, a fórmula usa `idx = 61`. O valor da geração 62 foi replicado até a geração 500, criando a impressão visual de que o algoritmo rodou 500 gerações em linha reta.
3. **Por que o Optuna não superou o Default?**
   * **Efeito Saturação (Ceiling Effect) no dataset Small:** Para instâncias de 16 tarefas, o espaço PFS é pequeno o suficiente para que 50.000 avaliações saturem a capacidade de busca. Tanto a configuração padrão quanto a sintonizada atingem o mesmo patamar de qualidade.
   * **Overfitting de Hiperparâmetros:** O Optuna foi executado em apenas 1 instância ([I_2_16_5_1.txt](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/data/Small/I_2_16_5_1.txt)), com somente 20 trials e 3 seeds, mas foi avaliado em 60 instâncias com 2, 3, 4 e 5 máquinas. O comportamento de instâncias de 2 máquinas é substancialmente diferente de instâncias com 5 máquinas.

---

## 3. O problema do Raio de Convergência no SCS

A versão SCS possui deficiências estruturais na etapa de clustering, documentadas nas próprias notas do aluno em [BRKGA_SCS.ipynb:1-6](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/notebooks_alunos/BRKGA_SCS.ipynb#L1-L6):

1. **A Maldição da Dimensionalidade no Raio Dinâmico:**
   Em [brkga_scs_solver.py:283](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/src/solvers/brkga_scs_solver.py#L283), a fórmula adotada é:
   $$R = \frac{1}{2 \cdot K^{1/N}}$$
   Com $N = 16$ (número de tarefas), a raiz 16ª de $K$ varia minimamente: para $K = 1$, $R = 0.50$; para $K = 10$, $R = 0.43$. O raio fica praticamente estático e excessivamente grande para o hipercubo unitário, agrupando indivíduos não correlacionados no mesmo cluster.
2. **Distância Euclidiana vs Chebyshev:**
   O SCS original previa a distância de Chebyshev ($L_\infty$) para definir vizinhanças hipercúbicas em chaves aleatórias, mas na implementação de [brkga_scs_solver.py:276](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/src/solvers/brkga_scs_solver.py#L276) foi usada a norma Euclidiana ($L_2$), que distorce o agrupamento em espaços multidimensionais.
3. **Ausência de Cooldown:**
   Diferente do protótipo com memória de estado do aluno, o solver atual não possui mecanismo de cooldown com decaimento geométrico. Se uma região tem bom fitness, ela dispara Hooke & Jeeves repetidamente na mesma bacia, esgotando o orçamento em pouquíssimas gerações.

---

## 4. Falhas e Inconsistências Críticas Detectadas no Código

### A. Telemetria do `cfo_count` no CSV está gravando sempre 2
No notebook [brkga_foco_v2.ipynb](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/brkga_foco_v2.ipynb):
```python
'cfo_count': int(result.engine_cfo_count)
```
Em [adapter.py:81-82](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/src/solvers/adapter.py#L81-L82), `engine_cfo_count` mede apenas a chamada final da função `_avaliar_na_engine` (1 para makespan + 1 para tardiness = 2). As dezenas de milhares de avaliações feitas durante a busca estão em `result.n_evaluations`, mas foram ignoradas no CSV.

### B. Dessincronização entre Espaço PFS e NPFS
Em [brkga_scs_solver.py:323](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/src/solvers/brkga_scs_solver.py#L323), a variável `best_fitness_npfs` só é atualizada se a busca local encontrar um valor menor que a chamada anterior. 
Se o BRKGA macro no espaço PFS encontrar uma solução com makespan menor (ex.: 919) e o refinamento NPFS só atingiu 922, o solver retorna 922! Como toda solução PFS pode ser replicada no NPFS com idêntico custo, `best_fitness_npfs` deveria ser mantido sempre como $\min(\text{best\_pfs}, \text{best\_npfs})$.

### C. Desperdício de Avaliações com Elitismo
Em [brkga_scs_solver.py:347](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/src/solvers/brkga_scs_solver.py#L347):
```python
populacao = nova_pop
fitness_pfs = np.array([avaliar_pfs(ind) for ind in populacao])
```
Os $N_{elite}$ indivíduos copiados da geração anterior são reavaliados a cada geração, gastando cerca de 20% do orçamento de avaliações do algoritmo sem necessidade.

---

## 5. Resumo da Análise Global do Projeto

1. **[principal.ipynb](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/principal.ipynb) (Experimento 1):**
   * Avaliou ACO, BRKGA, Neuro-BOA e PBIL-Fuzzy.
   * O **BRKGA foi o melhor candidato**, confirmando sua escolha. O Neuro-BOA provou ser inviável pelo tempo de execução de redes neurais a cada geração (40s a 46s por execução contra ~1.7s do BRKGA). O ACO convergiu prematuramente com gaps acima de 200%.
2. **[brkga_foco_v2.ipynb](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/brkga_foco_v2.ipynb) (Experimentos 2 e 3):**
   * A comparação de 100 gerações em [resultados_brkga_TODOS_100_gerac.csv](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/resultados/resultados_brkga_TODOS_100_gerac.csv) comprovou o poder da abordagem two-phase (PFS + transferência NPFS), mas mascarou o fato de que os concorrentes operavam em um espaço com 5x mais variáveis.
   * O teste de 500 gerações em [resultados_brkga_scs_optuna_500_gerac.csv](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/resultados/resultados_brkga_scs_optuna_500_gerac.csv) demonstrou que, para instâncias Small, o algoritmo atinge o teto em ~50 a 60 gerações. O Optuna não trouxe ganho estatisticamente significativo porque o espaço de 16 tarefas já estava saturado.

---

## Recomendações Práticas para os Próximos Passos

1. **Substituir o critério de parada:**
   Em vez de fixar `n_gen=500` com corte brusco em 50.000 avaliações, defina um orçamento explícito de CFOs (ex.: $10.000$ CFOs para comparações justas de curto prazo, e $30.000$ para longo prazo) e meça a convergência em função do **número real de CFOs**, e não de gerações nominais.
2. **Corrigir a telemetria do CSV:**
   Salvar `result.n_evaluations` no campo `cfo_count` em vez de `result.engine_cfo_count`.
3. **Recuperar o Stateful CS do aluno:**
   Implementar a distância de Chebyshev ($L_\infty$), contração/expansão de raio no sucesso/fracasso e o temporizador de cooldown do [`StatefulClusteringSearch`](file:///home/ronaldsc/UFMA/LACMOR/FlowShopScheduling/notebooks_alunos/BRKGA_SCS.ipynb#L6).
4. **Testar em Instâncias Médias/Grandes (Taillard):**
   As instâncias de 16 tarefas do dataset `Small` são pequenas demais para diferenciar meta-heurísticas refinadas. Para testar o verdadeiro impacto do Optuna e do SCS, o teste deve ser executado nas instâncias maiores de Taillard ($50 \times 10$, $100 \times 10$, $200 \times 10$).