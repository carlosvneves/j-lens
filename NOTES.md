# NOTES — Global Workspace / Jacobian Lens aplicado à colusão algorítmica

Fonte: *Verbalizable Representations Form a Global Workspace in Language Models*
(Transformer Circuits, 2026) + repo local `jacobian-lens/` (pacote `jlens`).

## 1. O problema que o artigo resolve

Queremos ler o que o modelo "pensa" **dentro** da rede, camada por camada, antes
de ele emitir o token de resposta. A ferramenta clássica é o **logit lens**;
o artigo propõe uma correção, o **Jacobian lens**.

### Logit lens (baseline)

O transformer mantém um *residual stream* `h_ℓ ∈ R^d` em cada camada ℓ.
O logit lens aplica a matriz de unembedding direto na ativação intermediária:

```
logit_lens_ℓ(h) = W_U · LN(h_ℓ)
```

Suposição implícita: toda camada usa o **mesmo sistema de coordenadas** da
camada final. Isso é falso nas camadas iniciais/médias — as representações
rotacionam e reescalam ao longo da profundidade. Resultado: leituras de camadas
iniciais são ruído.

### Jacobian lens (a correção)

Antes de desembeddar, transporta `h_ℓ` para a base da camada final usando o
**Jacobiano médio**:

```
J̄_ℓ = E[ ∂h_final / ∂h_ℓ ]          (média sobre ~1000 prompts de wikitext,
                                      todas as posições fonte e alvo)

jlens_ℓ(h) = W_U · LN( J̄_ℓ · h_ℓ )
```

`J̄_ℓ` é o **efeito causal linearizado médio** de uma direção na camada ℓ sobre
os logits finais. Duas consequências importantes:

1. **Corrige a mudança de base** entre camadas — leituras de camadas médias
   passam a ser interpretáveis.
2. **A média sobre milhares de contextos cancela o uso específico** da ativação
   no prompt atual e isola a *disposição geral a verbalizar* aquele conceito.
   Por isso os tokens do J-lens são lidos como "pensamentos intermediários",
   não como predições do próximo token.

Prova causal do artigo: em "The animal that spins webs has ___ legs", o J-lens
de camada média lê **"spider"** (palavra ausente do prompt); trocar esse vetor
pelo de "ant" muda a resposta de 8 para 6.

No código: `lens.apply(model, prompt, layers=..., positions=..., use_jacobian=True)`
retorna `(lens_logits, model_logits, input_ids)`. Com `use_jacobian=False` é o
logit lens puro — **sempre rodar os dois lado a lado** (baseline/controle).

## 2. J-space e o "workspace global"

O conjunto de direções que o J-lens consegue verbalizar forma o **J-space**.
Propriedades que justificam a analogia com Global Workspace Theory:

- **Verbalizável** — cada direção corresponde a um token/conceito que o modelo
  saberia dizer.
- **Sob controle dirigido** — dá para ativar/steerar deliberadamente.
- **Usado em raciocínio interno** — guarda passos *intermediários*, não só outputs.
- **Seletivo** — só ~**6–10% da variância** dos vetores de conceito. A maior
  parte da representação vive FORA do J-space.

Consequência metodológica direta: **ausência de um token no J-lens é evidência
fraca; presença é o sinal forte.**

## 3. Os três regimes de profundidade (Qwen3.5-4B, 36 camadas)

| Regime | Camadas aprox. | O que o J-lens mostra |
|---|---|---|
| Inicial | L0–L12 (~0–33%) | Ruído/ilegível. **Não interpretar.** |
| **Workspace** | **L12–L33** (~33–92%) | Conceitos abstratos, persistentes, muitas vezes ausentes do prompt e do output. É aqui que "pensamentos" aparecem. |
| Motor | últimas 2–3 camadas | Colapsa no token de saída iminente; J-lens ≈ logit lens ≈ output. |

Regra de leitura: **divergência J-lens vs logit-lens em camada média = sinal**
(conceito abstrato ainda não em coordenadas de output). Convergência tardia é
esperada e não informativa.

## 4. Como isso mapeia na pesquisa de colusão

Hipótese em termos de workspace: o modelo representa
**"coordenação/acordo vs independência" como conceito de camada média**,
*antes* de comitar o verdict Yes/No.

Assinaturas que confirmariam:

1. **Conceito antes da resposta**: tokens tipo `collusion, cartel, agreement,
   conspiracy` no J-lens da banda L12–L33 para prompts `combine/discuss`,
   em camadas *anteriores* ao crossover Yes/No.
2. **Dissociação entre condições**: mesmos tokens fracos/ausentes no prompt
   `independently` (que deve mostrar `independent, coincidence, competition`).
3. **Crossover Yes/No**: a camada onde `logit(Yes) − logit(No)` muda de sinal
   localiza onde a decisão comita. Deve vir *depois* do pico dos conceitos.
4. **Controle logit-lens**: conceito no J-lens mas NÃO no logit-lens da mesma
   camada = conteúdo interno genuíno, não eco de token da superfície.

Refutariam: keywords só nas últimas 2 camadas (eco do output); mesmos tokens
nas 3 condições (sem dissociação); Yes/No comitado já em L0–L5 (gatilho
lexical raso em "same price").

## 5. Caveats operacionais

- **Só tokens de vocabulário único**: "price fixing", "tacit collusion",
  "Sherman Act" são invisíveis (multi-token). Usar proxies de 1 token.
- **Lens treinado em wikitext** — prompt jurídico é fora de distribuição para
  o *lens* (não para o modelo). Divergência absurda do logit-lens → desconfiar.
- **Chat template obrigatório**: Qwen3.5-4B é modelo instruct/reasoning.
  Prompt cru = continuação de texto, não resposta. Usar
  `tokenizer.apply_chat_template(..., add_generation_prompt=True,
  enable_thinking=False)` para o verdict nascer no primeiro token gerado.
- **Posição de leitura**: com generation prompt, `positions=[-1]` é a última
  posição do prompt formatado — os logits ali predizem o **primeiro token da
  resposta**. (O walkthrough usa `-2` porque o prompt de lá termina em
  "...boot is" — caso diferente.)
- **Token IDs**: BPE distingue `"Yes"` de `" Yes"`. Verificar empiricamente
  qual variante o modelo emite antes de confiar em qualquer curva de logit.
- **Top-K do Qwen vem poluído** de pontuação/CJK — o artigo usa máscara/gloss
  (`assets/qwen_gloss.json.gz`). Sem filtro, você lê pontuação como conceito.
- **Correlação ≠ causação**: o lens é leitura. Teste forte = swap/patching do
  vetor J-lens (método spider→ant), com ~54–70% de sucesso reportado no artigo
  — esperar efeito ruidoso.
- **n=1 não sustenta nada**: bateria de paráfrases por condição antes de
  qualquer conclusão; reportar rank *relativo entre condições*, nunca presença
  absoluta (induction heads copiam tokens do prompt para camadas médias).

## 6. Swap: intervenção causal de primeira ordem

O `run_local.py` inclui uma versão do experimento spider→ant do artigo,
no domínio da colusão. Em vez de trocar o vetor lido pelo lens, injetamos no
residual da camada L a direção que o Jacobiano diz empurrar um conceito:

```
Δh = α · J̄_Lᵀ · (w_collusion − w_independent)
```

Racional: `J̄_L ≈ ∂(logits finais)/∂h_L`, então `J̄_Lᵀ · w_token` é a direção
em `h_L` que mais aumenta o logito daquele token (gradiente de primeira
ordem). Somar a diferença sobe `collusion` e desce `independent` de uma vez.
Se a resposta do prompt "independently" virar No→Yes com α moderado, o
conceito é **causa** do verdict, não correlato. α grande demais que quebra a
fluência do texto = intervenção gritando por cima do modelo, descartar.

## 7. Próximos passos

1. **[após o teste local] Implementar a visualização** — dashboard do
   `jlens.vis` (`compute_slice` + `build_page` + gloss
   `assets/qwen_gloss.json.gz`) para inspecionar todas as posições × camadas,
   não só a última posição.
2. Bateria de paráfrases (3 condições × ~10) com métrica
   `logit(Yes) − logit(No)` por camada + rank relativo das keywords.
3. Controles do red team: paráfrase sem palavra-gatilho, controle negativo
   ("discuss the weather").
4. Replicar no Qwen3.6-27B no Colab.
