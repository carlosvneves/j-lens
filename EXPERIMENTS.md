# EXPERIMENTS — diário de laboratório

Formato: condições experimentais → resultados → modificações → próxima run.
Conceitos e matemática: `NOTES.md`. Código: `run_local.py`.

---

## RUN-001 — 2026-07-13 — primeira leitura local (2 condições)

### Condições

- Modelo: Qwen3.5-4B bf16, MPS local. Lens: `qwen-n1000` (wikitext).
- Chat template, `enable_thinking=False`, leitura em `positions=[-1]`
  (prediz o 1º token da resposta). Geração greedy, 12 tokens.
- Prompts: `independent` ("independently choose the same price") e
  `collusion` ("combine to choose the same price"), ambos "...Illegal?
  Just answer Yes or No."
- Métricas: top-6 por camada (logit-lens e J-lens), `logit(Yes)−logit(No)`
  por camada, rank de {collusion, agreement, conspiracy, independent,
  cartel} no J-lens por camada.

### Resultados

- Respostas corretas: independent → 'No', collusion → 'Yes'. Margens finais
  pequenas (−0.75 e +0.25) — modelo quase indeciso nos dois casos.
- **Yes−No positivo nas camadas médias nas DUAS condições** (até +13 no
  independent). Interpretação: viés de formato do slot de resposta ("Yes" é
  genérico ali), não verdict. O sinal está no contraste: collusion
  consistentemente mais "Yes" em toda a banda workspace, gap crescente
  (Δ = +1.1 em L16 → +4.6 em L28).
- **Crossover tardio** no independent: +5.25 (L28) → −2.62 (L30). O verdict
  comita nas últimas 2 camadas (regime motor), como o artigo prevê.
- **"depends/whether" dominam L18–L22** no independent (mais fraco no
  collusion): o modelo representa condicionalidade — estrutura jurídica
  correta (paralelismo consciente legal, acordo ilegal). Achado.
- **Dissociação mais forte: keyword `independent`**, não `collusion`:
  rank 736–1600 (top ~0.5%) em quase todas as camadas no prompt
  independent; despenca para 70k–100k em L16–L26 no prompt collusion.
  O modelo representa/suprime "independência" conforme a condição.
- `conspiracy` direção certa (961 vs 1722 em L28) mas em regime motor —
  evidência fraca. `collusion/cartel` ranks altos nos dois — esperado
  (J-space = 6–10% da variância; lens é wikitext).
- Lixo em L0–L10 (`<|endoftext|>`, `<|im_end|>`): posição -1 é fim do
  template; lens fora de distribuição para tokens especiais. Ignorado.
- Swap: run abortou antes (bug de formato do hook, corrigido) — swap ainda
  sem resultado registrado.

### Ameaças à validade identificadas

1. Eco lexical: "independently"/"combine" estão no prompt; induction heads
   podem explicar os ranks sem conceito interno.
2. Viés do slot Yes/No contamina o nível absoluto da curva.
3. n=1 por condição.

### Modificações para RUN-002

- (a) Filtro de tokens especiais no top-K.
- (b) Tabela automática Δ(Yes−No) vs baseline `independent`.
- (c) Tabela de rank por keyword × condição (contraste direto) com
  percentil do vocabulário.
- Nova condição `no_echo`: coordenação descrita SEM palavra-gatilho
  ("After a private phone call between their sales managers...").
- Nova condição `open_concept`: sem eco E sem slot Yes/No — pergunta aberta
  "What is the one-word legal term for this practice?" (verdict = o próprio
  conceito).
- Dashboard jlens por prompt (`viz_<label>.html`), com keywords + Yes/No
  pinados (rank visível como curva em todas posições × camadas).

---

## RUN-002 — pendente — controles de eco e de viés de formato

### Mudança forçada de modelo (2026-07-18)

`Qwen/Qwen3.5-4B` sumiu do HF Hub (404); cache local apagado. Novo par
modelo+lens: **google/gemma-4-E4B-it** (4B denso, multimodal, bf16 MPS) +
lens `solarkyle/jspace-lenses :: gemma-4-e4b-it/lens.pt` (fit: 100 prompts
WikiText-103, bf16). Consequências:

- RUN-002 deixa de ser diretamente comparável à RUN-001 (modelo diferente,
  vocabulário diferente, lens com n=100 vs n=1000). As 4 predições abaixo
  são agnósticas de modelo e continuam valendo; contrastes são sempre
  intra-run.
- Se os achados da RUN-001 (dissociação de `independent`, Δ crescente)
  reaparecerem no Gemma, vira evidência de robustez entre famílias — de
  graça.
- Código agora multi-modelo: registro em `model_configs.py`, seleção via
  `JSPACE_MODEL` (default `gemma-4-e4b-it`; `qwen3.5-4b` mantido para
  reprodutibilidade; `qwen3.6-27b` já configurado para o Colab, lens do
  mesmo registro solarkyle).
- Sem gloss para Gemma (o gloss local é do vocab Qwen) — dashboards
  funcionam sem tradução de tokens raros. `viz_*.html` agora inclui o
  modelo no nome do arquivo.
- Gemma exige aceitar licença no HF + token de leitura.

### Condições

RUN-001 + as modificações acima, agora em gemma-4-E4B-it. 4 condições:
independent, collusion, no_echo, open_concept. Baseline dos deltas:
independent.

### Predições (registradas ANTES da run)

1. `no_echo`: se o sinal de RUN-001 é conceito e não eco, a coluna
   Δ(Yes−No) do no_echo deve ser positiva e crescente na banda workspace,
   como a do collusion — sem nenhuma palavra-gatilho no prompt.
   Se Δ ≈ 0, o resultado de RUN-001 era eco lexical.
2. `no_echo`: rank de `independent` deve despencar (como no collusion),
   apesar de a palavra não aparecer no prompt.
3. `open_concept`: rank de `collusion` (ou `cartel`) deve cair até próximo
   de 1 nas camadas finais (é a resposta), com emergência ANTES, na banda
   workspace (~L14–L26). Se só emergir nas 2 últimas camadas → conceito é
   output, não pensamento intermediário. Coluna Yes−No: ignorar (sem slot
   Yes/No neste prompt).
4. Swap (independent→collusion na L~24): resposta vira No→Yes com α
   moderado em pelo menos parte das runs (artigo: 54–70% de sucesso).

### Resultados

(preencher após a run)

---

## Backlog metodológico

- Bateria de paráfrases (~10 por condição), média ± desvio de Δ(Yes−No).
- Controle negativo: "discuss the weather and choose the same price".
- Patching completo de residual entre prompts (fidelidade total ao artigo,
  vs a aproximação de 1ª ordem do swap atual).
- Replicar no Qwen3.6-27B (Colab).

## Como ler "rank" (referência rápida)

O lens dá um logito para cada um dos ~151k tokens do vocabulário. Rank =
posição do token na fila ordenada por logito (1 = mais provável). A coluna
"top" das tabelas mostra apenas ranks 1–6 — keyword com rank 736 nunca
aparece lá e mesmo assim está à frente de 99.5% do vocabulário (por isso o
script agora imprime `736 (0.5%)`). Rank absoluto não é evidência (lens
wikitext, J-space parcial); contraste do rank da MESMA keyword entre
condições, na MESMA camada, é.
