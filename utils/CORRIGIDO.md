# Log de Correções (Flow Shop Scheduling)

Este documento centraliza e resume todas as correções implementadas no projeto para garantir integridade científica, concorrência justa e máxima performance metodológica.

---

## 1. Critério de Parada por Orçamento (CFOs em vez de Gerações)
**Status:** Concluído  
**Arquivos modificados:** `brkga_scs_solver.py`, `brkga_solver.py`, `adapter.py`, `brkga_foco_v2.ipynb`

**Problema original:**  
O código usava `n_gen` (número de gerações) como critério de parada. Porém, o BRKGA-SCS gasta centenas de avaliações extras por geração (via Hooke & Jeeves), enquanto o BRKGA padrão gasta apenas `tam_pop`. Usar gerações como medida gerava uma comparação injusta.

**O que foi corrigido:**
- O parâmetro `n_gen` / `max_generations` foi substituído por `max_evals` (orçamento estrito de CFOs) em todos os solvers e no adapter.
- O loop principal do BRKGA-SCS agora usa `while evals_count < max_evals_total` em vez de `for gen in range(max_generations)`.
- O notebook `brkga_foco_v2.ipynb` teve a constante `N_GEN` substituída por `MAX_EVALS = 50000`.

---

## 2. Correção da Telemetria (cfo_count = 2) no CSV
**Status:** Concluído  
**Arquivos modificados:** `brkga_foco_v2.ipynb`, `adapter.py`

**Problema original:**  
O CSV gravava `cfo_count = 2` para todas as execuções do BRKGA-SCS. Isso acontecia porque `engine_cfo_count` era calculado apenas sobre a chamada final de validação (`_avaliar_na_engine`), que fazia exatamente 2 avaliações (makespan + tardiness), ignorando as dezenas de milhares de avaliações internas do solver.

**O que foi corrigido:**
- O notebook agora grava `result.n_evaluations` (contagem real do solver) na coluna `cfo_count`.
- A função `amostrar_historico` foi reescrita: agora recebe uma lista de tuplas `(evals, makespan)` e amostra por marcos de CFOs (ex: a cada 5000 avaliações), em vez de por gerações fictícias.
- As colunas do CSV mudaram de `mks_gen20, mks_gen40, ...` para `mks_ev5000, mks_ev10000, ...`.

---

## 3. Restauração do Stateful Clustering Search
**Status:** Concluído  
**Arquivos modificados:** `brkga_scs_solver.py`

**Problema original:**  
O solver extraído do notebook original usava uma versão simplificada e incompleta do clustering (classe `ClusterStateful` com apenas 3 campos). Essa versão:
- Usava **Distância Euclidiana** (`np.linalg.norm`) em vez da Chebyshev.
- Tinha um **raio estático** calculado por `1/(2·K^{1/N})` que não funcionava bem para N=16.
- **Não tinha cooldown**: clusters estagnados continuavam disparando buscas locais, desperdiçando orçamento.
- O notebook original já tinha a versão correta no final (classes `ClusterState`, `Cluster`, `StatefulClusteringSearch`), mas essa versão nunca foi portada.

**O que foi corrigido (transplante do notebook original):**
- **Distância de Chebyshev (L∞):** Usando `np.max(np.abs(centroid - individual))` para delimitar vizinhança hipercúbica em [0,1]^D.
- **Raio Adaptativo Dinâmico:** Se busca local tem sucesso → contrai raio em 10% (`radius * 0.9`). Se falha → expande em 10% (`radius * 1.1`). Limitado entre `r_min=0.05` e `r_max=0.25`.
- **Cooldown Geométrico:** Se a busca local não melhora, o cluster entra em estado `COOLDOWN` por `base_cooldown × gamma^estagnacao` gerações (com limite `max_cooldown=50`). Enquanto em cooldown, o cluster **rejeita** novas buscas locais, economizando CFOs preciosos.
- **Enum de Estado:** Cada cluster agora tem um estado explícito: `ACTIVE`, `COOLDOWN` ou `EXHAUSTED`.
- **Limite de clusters:** Máximo de `num_clusters_max=5` clusters simultâneos.

---

## 4. Sincronização PFS ↔ NPFS (Makespan)
**Status:** Concluído  
**Arquivos modificados:** `brkga_scs_solver.py`

**Problema original:**  
O `best_fitness_npfs` só era atualizado quando a busca local (Hooke & Jeeves) encontrava melhoria. Se a evolução macro (PFS) encontrasse uma solução superior naturalmente, essa melhoria **não era propagada** para o NPFS. Resultado: o solver às vezes retornava um makespan NPFS pior do que o PFS.

**O que foi corrigido:**  
- Sempre que `best_fitness_pfs` melhora, o código agora verifica se é melhor que `best_fitness_npfs`. Se sim, projeta a solução PFS para NPFS via `np.tile` e atualiza ambos os rastreamentos.

---

## 5. Eliminação de Reavaliação de Elites
**Status:** Concluído  
**Arquivos modificados:** `brkga_scs_solver.py`

**Problema original:**  
Na virada de geração, o código fazia `fitness_pfs = np.array([avaliar_pfs(ind) for ind in populacao])`, reavaliando **todos** os indivíduos incluindo os `n_elite` elites que foram copiados intactos da geração anterior. Como o fitness deles já era conhecido, isso desperdiçava ~20% do orçamento.

**O que foi corrigido:**  
- Os elites são copiados com seu fitness já calculado (`novo_fitness[:n_elite] = fitness_pfs[:n_elite]`).
- Apenas os mutantes e filhos de crossover são avaliados.

---

## 6. Atualização do Adapter
**Status:** Concluído  
**Arquivos modificados:** `adapter.py`

**O que foi corrigido:**  
- A função `rodar_brkga_scs` no adapter foi atualizada para passar os novos parâmetros do Stateful Clustering (`num_clusters_max`, `r_min`, `r_max`, `base_cooldown`, `gamma`) em vez do antigo `raio_cluster`.
- O `n_evaluations` no `SolverResult` agora reflete o `total_evaluations` real retornado pelo solver.
