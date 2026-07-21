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

### Incidente de infraestrutura (2026-07-18)

`Qwen/Qwen3.5-4B` deu 404 no HF Hub por algumas horas; cache local (8.8 GB)
foi apagado e o código foi migrado para gemma-4-E4B-it + lens do registro
`solarkyle/jspace-lenses`. O modelo VOLTOU ao Hub no mesmo dia; default
revertido para `qwen3.5-4b` — RUN-002 segue comparável à RUN-001 (mesmo
modelo, mesmo lens n=1000, agora servido da branch main do
neuronpedia/jacobian-lens, sem revision especial). Requer re-download do
modelo (~9 GB).

Saldo positivo do incidente — código agora é multi-modelo:

- Registro em `model_configs.py`, seleção via `JSPACE_MODEL`
  (default `qwen3.5-4b`).
- `gemma-4-e4b-it` configurado como alternativa local: fallback se o Qwen
  sumir de novo E teste de robustez entre famílias (mesma bateria, outra
  família/vocabulário). Lens do solarkyle (n=100) porque o neuronpedia só
  tem o E4B base, não o -it.
- `qwen3.6-27b` pronto para o Colab com lens n=1000 do neuronpedia
  (fonte primária unificada).
- `viz_*.html` agora inclui o modelo no nome do arquivo.

### Condições

RUN-001 + as modificações acima, mesmo modelo da RUN-001 (Qwen3.5-4B).
4 condições: independent, collusion, no_echo, open_concept. Baseline dos
deltas: independent.

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

### Resultados (run de 2026-07-18)

Respostas do modelo: independent → 'No', collusion → 'Yes', no_echo →
'Yes', open_concept → 'Conspiracy'. Relatório didático:
`relatorio_run002.html`.

**P1 — CONFIRMADA.** Δ(Yes−No) vs baseline: ruído (±0.4) até L12 nas duas
colunas; a partir de L14 ambas positivas e crescentes na banda workspace —
collusion +1.06 (L16) → +4.62 (L28); no_echo +0.56 (L16) → +5.38 (L28).
no_echo acompanha collusion sem palavra-gatilho → sinal de RUN-001 não era
eco lexical.

**P2 — CONFIRMADA.** Rank de `independent` no no_echo salta de ~800 (L14)
para 18k–76k em L16–L26 (pico 76k em L18), espelhando collusion (32k–100k),
enquanto no baseline fica em 736–20k. Dissociação começa exatamente em L16
(início da banda) nas duas condições de coordenação.

**P3 — CONFIRMADA (resultado mais forte).** open_concept: `collusion` cai
de 146k (L16) → 3.2k (L18) → **rank 1 em L20–L28** — emergência no meio da
banda workspace, ~8 camadas antes do regime motor. `cartel` (12→2) e
`conspiracy` (38→3) acompanham; `agreement` chega a rank 20 em L24.
`independent` é suprimido ATIVAMENTE: rank 203k–227k em L18–L26, pior que
o acaso (~124k) → sugere eixo bipolar coordenação↔independência.
Nota: o modelo respondeu 'Conspiracy' (não 'Collusion') apesar de
`collusion` rank 1 no lens até L28 — na última camada o top vira
'Con/Coll/Cart' (fragmentos BPE de início de palavra; a resposta começa
turno novo, sem espaço à frente).

**P4 — FALHOU.** Swap independent→collusion na L21, α ∈ {4, 8, 16}:
resposta permanece 'No' em todos. A intervenção FUNCIONA no lens
(J-lens@L22 mostra 'collusion' no top-1 com α≥8, e com α=16 o top é só
variantes de collusion), mas o verdict não vira. Leitura: o conceito foi
escrito no workspace, porém (a) direção de 1ª ordem via J̄ᵀ pode não ser a
direção causal usada downstream, (b) delta somado em TODAS as posições
pode diluir/conflitar, (c) artigo reporta 54–70% de sucesso — n=1 pode ser
o azar da moeda. Diagnóstico pendente antes de concluir "sem causalidade".

### Modificações pós-run

- Output completo agora vai para `report_<modelo>_<timestamp>.txt`;
  console só progresso (pedido do usuário; report_*.txt no .gitignore).

### Próximos passos sugeridos para RUN-003 (swap)

- Variar camada do swap (L16–L26), α intermediários, delta só na última
  posição vs todas, e/ou patching real de residual entre prompts
  (substituição, não soma) — o full patching do backlog é o teste decisivo.

---

## RUN-003 — pendente — diagnóstico causal do swap

### Motivação

P4 da RUN-002 falhou de forma informativa: o conceito foi escrito no
subespaço legível (J-lens@L22 lê 'collusion' com α≥8) mas o verdict não
virou. RUN-003 isola POR QUÊ, variando os 3 suspeitos de uma vez:
camada, escopo posicional da intervenção e tipo de intervenção
(direção de 1ª ordem vs patching real de residual).

### Condições

Modelo: Qwen3.5-4B (mesmo da RUN-001/002). Prompt alvo: `independent`
(verdict baseline 'No'). Doador do patch: `collusion`. Script:
`run_swap003.py` (output completo em report_*.txt, console só progresso).

- **Braço A — steering de 1ª ordem (grade):** camadas {16,18,20,22,24,26}
  × α {4,8,16} × escopo {todas as posições, só última posição do prompt}.
  Métrica por célula: resposta gerada + margem final logit(Yes)−logit(No).
- **Braço B — patching real (teste decisivo):** substituir (não somar) o
  residual da última posição do prompt independent pelo residual da última
  posição do prompt collusion, camada a camada {14,16,18,20,22,24,26,28}.
  Sem α: substituição é tudo-ou-nada. Mesmas métricas.
- Intervenção só no prefill (posições do prompt); passos de decode não são
  tocados — o efeito propaga via KV cache.

### Predições (registradas ANTES da run)

1. **Braço B vira o verdict** (No→Yes) em pelo menos uma camada da banda
   L16–L26. Se NEM o patching real virar, o verdict não depende
   causalmente do residual da última posição nessa banda (sinal
   distribuído em outras posições) — redesenhar para patch multi-posição.
2. **Braço A, escopo última posição ≥ escopo todas**: restringir o delta à
   posição de leitura reduz conflito com o contexto; esperamos margens
   Yes−No mais deslocadas (ou flip) no escopo 'última'.
3. **Margem move monotonicamente com α** dentro de cada (camada, escopo)
   mesmo sem flip; se a margem não se mexer, a direção J̄ᵀ é ortogonal ao
   circuito do verdict (leitura ≠ escrita) — resultado publicável por si.
4. Camadas mais eficazes: meio da banda (L18–L24), onde a RUN-002 mostrou
   o conceito estável; L26+ tarde demais (verdict já comitado no motor).

### Resultados (run de 2026-07-19, report_qwen3.5-4b_swap003_20260719-221237.txt)

Baseline: 'No', margem Yes−No = −0.75.

**P1 — CONFIRMADA (com nuance).** Braço B (patching real) FLIPOU em L22:
'Yes', margem +0.12. Estrutura da curva: Δ margem = 0.00 em L14–L16,
+0.62 em L18, +0.75/+0.88 em L20–L28. O flip só ocorre em L22 porque a
margem patchada satura em ~0.00/+0.12 — exatamente o nível do próprio
doador (collusion tinha margem final +0.25 na RUN-002). Ou seja: o patch
da última posição TRANSFERE o estado de decisão do doador quase por
inteiro a partir de L18; o modelo doador é que é quase indeciso.
Verdict depende causalmente do residual da última posição na banda.

**P2 — REFUTADA.** Escopo 'última' ≈ 'todas' no braço A: ambos nulos.
Não era conflito posicional.

**P3 — CONFIRMADA a hipótese nula (resultado central da run).** Steering
J̄ᵀ NÃO move a margem em direção a Yes em nenhuma célula da grade
(6 camadas × 3 α × 2 escopos = 36 células, zero flips; margens −0.4 a
−1.2, i.e. imóveis ou levemente piores). Com α=16: L18/todas degrada a
margem (−4.0) e L26/todas faz o modelo cuspir 'collus' como resposta —
gritar o token, não mudar o julgamento. Conclusão: a direção de leitura
do J-lens é ORTOGONAL ao canal causal do verdict — **leitura ≠ escrita**.
O lens lê o workspace; escrever nele exige o estado real (patch), não a
direção de 1ª ordem do unembedding.

**P4 — PARCIAL.** Camada eficaz do patch: L22 (meio da banda, como
previsto), mas o efeito na margem já satura em L18–L20 e persiste até
L28 (sem janela que se fecha no motor — possivelmente porque o patch na
última posição carrega o verdict já formado do doador).

### Leitura conjunta RUN-002 + RUN-003

O par (P4 da RUN-002, P3 da RUN-003) forma um resultado coerente e
publicável: o subespaço que o J-lens LÊ contém o conceito (collusion
legível, rank 1 no open_concept), mas empurrar o residual ao longo de
J̄ᵀw não altera a computação downstream — enquanto transplantar o estado
completo altera (flip em L22). Workspace verbalizável = janela de
observação; o controle causal mora no complemento (90-94% da variância
que o lens não captura).

---

## RUN-004 — pendente — sair do n=1: paráfrases, controle negativo, doador forte

### Motivação

Todos os achados até aqui são n=1 por condição, e o flip do patch (RUN-003)
está no fio da navalha porque o doador é quase indeciso (margem +0.25).
RUN-004 dá variância a tudo: bateria de paráfrases, controle negativo de
"contato inócuo" e patch com doador de margem folgada.

### Condições

Modelo: Qwen3.5-4B. Script: `run_004.py` (output → report_*.txt).

- **Braço A — bateria de paráfrases (n=6 por condição):** independent,
  collusion, no_echo — 6 formulações cada, mesmo formato "...Illegal?
  Just answer Yes or No.". As paráfrases do no_echo não contêm NENHUMA
  palavra-gatilho (agree/collude/cartel/conspire/coordinate/combine/
  pact/fix). Métricas por condição: resposta gerada (contagem Yes/No),
  Δ(Yes−No) por camada média ± desvio vs média do independent, rank de
  'independent' (mediana e IQR) na banda L18–L26.
- **Braço B — controle negativo (n=3):** contato explícito + conteúdo
  inócuo (tempo/golfe/caridade) + preços iguais. Mede se QUALQUER
  contato entre concorrentes dispara o sinal de coordenação.
- **Braço C — patch com doador forte:** doador = cartel explícito
  ("form a secret cartel agreement to fix prices..."); reportar margem
  do doador; patch da última posição nas camadas {18,20,22,24,26} sobre
  o prompt independent canônico.

### Predições (registradas ANTES da run)

1. Respostas: collusion e no_echo → 'Yes' em ≥5/6 paráfrases;
   independent → 'No' em ≥5/6.
2. Δ(Yes−No) médio na banda L16–L28: collusion e no_echo positivos com
   média − desvio > 0 (separação de ~1σ do baseline); as duas condições
   estatisticamente indistinguíveis entre si (conceito, não gatilho).
3. Rank de 'independent' em L18–L26: mediana do independent < 20k;
   mediana de collusion/no_echo > 40k; IQRs sem sobreposição.
4. Controle negativo fica entre independent e no_echo, mais perto do
   independent (Δ médio < metade do Δ do no_echo). Se colar no no_echo,
   o sinal é "contato + preço igual", não coordenação inferida —
   downgrade importante da interpretação.
5. Doador forte: margem própria > +2; patch flipa em ≥3 das 5 camadas
   com margem final > +1 (flip robusto, não fio de navalha).

### Resultados (run de 2026-07-19, report_qwen3.5-4b_run004_20260719-222245.txt)

**P1 — PARCIAL.** independent 6/6 'No' ✔; collusion 5/6 'Yes' ✔ (a exceção
é notável: "agree to charge the same price" → 'No', margem −0.50 — acordo
explícito e o modelo erra o verdict!); **no_echo 2/6 'Yes' ✘** (previa
≥5/6). Ver dissociação abaixo — o miss é informativo, não ruído.

**P2 — CONFIRMADA (com gradiente).** Δ(Yes−No) vs média do independent na
banda: collusion +1.07±0.40 (L16) → +9.65±2.13 (L28); no_echo +0.71±0.35
→ +5.27±3.41. Média−desvio > 0 em toda a banda nas duas ✔. MAS não são
indistinguíveis: collusion ≈ 2× no_echo consistentemente — o workspace
codifica a FORÇA da evidência (explícita > inferida), não um binário.

**P3 — CONFIRMADA na direção, limiar raspado.** Mediana do rank de
'independent' em L18–L26: independent 21.8k (previa <20k — raspou),
collusion 86.9k ✔, no_echo 48.3k ✔ (>40k). Min–max se sobrepõem
(independent chega a 61k; no_echo mínimo 19.5k) — separação de medianas
clara, mas não categórica com n=6.

**P4 — CONFIRMADA.** neg_control fica perto do independent: Δ na banda
+0.14 a +2.04, sempre < metade do no_echo; rank de 'independent' mediana
26.2k (≈ o próprio independent, 21.8k). Contato inócuo + preços iguais
NÃO dispara o sinal — o que o workspace representa é coordenação
INFERIDA, não mera co-ocorrência de contato e paralelismo. (Há um leve
gradiente residual em L24–L28, +1.2 a +2.0: "contato" deixa um traço
fraco de suspeita.)

**P5 — CONFIRMADA.** Doador forte: margem +2.12 (>+2 ✔). Patch flipa
**5/5 camadas** (L18–L26), margens +0.50 a +1.12 (4/5 ≥ +1.0). O flip da
RUN-003 não era fio de navalha do método — era a margem fraca do doador.
Transferência parcial (~metade da margem do doador), consistente com o
patch de UMA posição.

### Achado novo (não previsto): suspeita interna sem condenação

No no_echo o sinal interno é claramente pró-Yes (P2 ✔) mas o verdict
final é 'No' em 4/6 — margens finais todas perto de zero (−1.5 a +0.9).
O modelo REPRESENTA a suspeita no workspace e não comete a condenação no
output. Paralelo jurídico direto: evidência de contato + paralelismo
gera inferência mas não basta para condenar (doutrina de conscious
parallelism / plus factors). Para detecção de colusão algorítmica é o
resultado-chave da run: o monitor interno (lens) vê o que o output
esconde — ranking de risco interno ≠ resposta verbalizada.

Escada dose-resposta completa (Δ médio na banda L22–L28):
independent 0 < neg_control ~+1.4 < no_echo ~+4.2 < collusion ~+8.0.

---

## RUN-005 — replicação cross-model: gemma-4-e4b-it

### Motivação

Todo achado até aqui (RUN-002 a RUN-004) é n=1 modelo (Qwen3.5-4B).
Rodar as mesmas três baterias (RUN-002/003/004) no gemma-4-e4b-it —
outra família, outro tokenizer, mesmo porte (~4B) — pra saber se os
achados são propriedade do method ou do modelo específico.

Nota de config: `gemma-4-e4b` (base) não tem chat template — falhou de
cara. Usado `gemma-4-e4b-it` (instruction-tuned), lens do solarkyle
(fit n=100, não n=1000 do neuronpedia — [[j-space-research-context]]).

### Condições

Reexecução idêntica de `run_local.py`, `run_swap003.py`, `run_004.py`
com `JSPACE_MODEL=gemma-4-e4b-it`, sem alterar código nem baterias.

### Resultados

**Escala de margem diferente.** Baseline Yes−No do Gemma é −8.38 (vs
−0.75 do Qwen) — Gemma é muito mais decidido/conservador no 'No' cru.
Margens absolutas não são comparáveis entre modelos; só sinal e padrão
relativo.

**RUN-002 (baseline) replica bem no open_concept.** Pergunta aberta
("one-word legal term") converge fortemente pra 'collusion'/'cartel'/
'antitrust' em L28–L40, com rank 1 (0.0%) em L38 — sinal até mais nítido
que no Qwen. Verdict do collusion prompt fica ambíguo ("Whether... 
depends") em vez de 'Yes' — Gemma hedgeia mais que Qwen.

**RUN-003 — achado central NÃO replica, e inverte.**
No Qwen: steering (Braço A) nunca flipava; patching (Braço B) flipava em
L22. No Gemma acontece o OPOSTO:
- Braço A (steering): **FLIP em L22, α=8, escopo='todas'** (margem
  +6.4) — steering funcionou aqui, ao contrário do Qwen.
- Braço B (patching real): **nenhum flip** em nenhuma das 8 camadas,
  apesar do Δ vs baseline crescer até +6.75 em L28 — patch NÃO foi
  suficiente pra virar o verdict, ao contrário do Qwen (flip em L22).
- Conclusão: "leitura ≠ escrita" (RUN-003/Qwen) não é uma propriedade
  universal do método — é dependente de arquitetura/modelo. Em outro
  modelo o canal causal pode estar mais alinhado com a direção J̄ᵀw
  (steering funciona) ou o subespaço patchado pode não ser suficiente
  (patching falha). As duas descobertas sobre causalidade da RUN-003 são
  específicas do Qwen, não do fenômeno.

**RUN-004 — achados parciais, mais fracos que no Qwen.**
- P1: independent 6/6 'No' ✓. collusion 4/6 'Yes' (Qwen: 5/6) — mais
  hedge ("Whether... combining... choose"). no_echo **0/6 'Yes'** (Qwen:
  2/6) — verdict do Gemma é ainda mais conservador que o do Qwen aqui.
- P2 (dose-resposta): collusion tem Δ positivo crescente na banda
  L16–L28 (+0.71 a +2.25, salto pra +16 em L30+ fora da banda
  original). no_echo NÃO é robustamente positivo na mesma banda —
  oscila entre negativo e levemente positivo (-1.63 a +0.87) — o sinal
  interno de "suspeita sem condenação" do Qwen não aparece com a mesma
  clareza aqui.
- P3 (separação de rank): muito mais fraca. independent mediana 13977,
  collusion 19854, no_echo 20846, neg_control 34384 — faixas quase
  todas sobrepostas (Qwen tinha separação de ordem de grandeza:
  independent 21.8k vs collusion 86.9k). neg_control ficou com rank
  MAIOR que collusion/no_echo — na direção oposta da hipótese.
  neg_control também respondeu quase igual ao no_echo em texto
  verbalizado (ambos majoritariamente 'No' com 1 hedge).
- P5 (patch com doador forte): doador margem +10.00 (>+2 ✔), mas só
  **2/5 flips** (L24, L26) — vs 5/5 do Qwen. Transferência causal mais
  fraca / mais tardia nesse modelo.

**Ressalva de banda.** `RANK_BAND`/`STEER_LAYERS`/`PATCH_LAYERS` foram
copiados literalmente do Qwen (n_layers≈36) pro Gemma (n_layers=42) sem
reajustar proporcionalmente — a banda "workspace" pode estar desalinhada
em profundidade relativa entre os dois modelos. Isso pode explicar parte
da atenuação dos efeitos no Gemma, não só diferença real de arquitetura.

### Síntese

O método (Jacobian Lens) generaliza bem pra **leitura de conceito**
(open_concept, dose-resposta qualitativa de collusion). Não generaliza
automaticamente pros achados **causais e quantitativos finos** da
RUN-003/004 (qual intervenção funciona, separação de rank, robustez do
sinal de no_echo) — esses são sensíveis a arquitetura/modelo e exigiriam
recalibrar bandas de camada e re-registrar predições por modelo, não
assumir que valem cross-family.

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
