# Run local (MacBook / MPS) — Jacobian Lens em prompts de colusão.
#
# COMO RODAR
# ----------
# 1. Instalar dependências (uma vez), a partir da raiz do projeto j-space:
#      uv add ./jacobian-lens transformers accelerate
#    (instala o pacote jlens local + HF transformers; torch vem junto)
#
# 2. Login no Hugging Face: Gemma exige aceitar a licença na página do modelo
#    e token com leitura (`hf auth login` ou `export HF_TOKEN=...`).
#
# 3. Executar (modelo default: gemma-4-e4b-it, ver model_configs.py):
#      uv run python run_local.py
#    Outro modelo do registro:
#      JSPACE_MODEL=qwen3.5-4b uv run python run_local.py
#
#    Primeira execução baixa o modelo (~8 GB p/ E4B) + lens (~0.5 GB) para
#    ~/.cache/huggingface. Execuções seguintes usam o cache.
#    Tempo esperado no M5: 1-2 min de load + segundos por prompt.
#
# 4. Requisitos: ~10 GB livres de RAM unificada (modelo em bf16), macOS com
#    MPS (Apple Silicon). Sem GPU cai em CPU — funciona, só mais lento.
#
# Exemplo didático, sem dashboard. Para cada prompt:
#   1. formata com chat template (thinking desligado → verdict = 1º token gerado)
#   2. lê logit-lens e J-lens em cada camada fitted, na última posição do prompt
#   3. imprime top tokens por camada, curva logit(Yes)-logit(No), rank de keywords
#   4. gera a resposta real do modelo para conferência
#
# OUTPUT: as tabelas completas vão para report_<modelo>_<timestamp>.txt;
# o console mostra só o progresso (1 linha por etapa) e o caminho do txt.
#
# Conceitos: ver NOTES.md.

# %%%
import datetime
import os
import sys

import torch
import transformers

import jlens

from model_configs import DEFAULT_MODEL, MODEL_CONFIGS

ACTIVE_MODEL = os.environ.get("JSPACE_MODEL", DEFAULT_MODEL)
CFG = MODEL_CONFIGS[ACTIVE_MODEL]
print(f"modelo ativo: {ACTIVE_MODEL} ({CFG['model_id']})")

# Dois exemplos do test.txt: a única diferença é "independently" vs "combine".
# Se o modelo representa "coordenação" como conceito interno, a diferença deve
# aparecer nas camadas médias ANTES do verdict.
PROMPTS = {
    "independent": "Two competitors independently choose the same price. Illegal? Just answer Yes or No.",
    "collusion": "Two competitors combine to choose the same price. Illegal? Just answer Yes or No.",
    # RUN-002: controle de ECO LEXICAL. Descreve coordenação sem nenhuma
    # palavra-gatilho (combine/collude/agree/independent). Se os sinais de
    # camada média persistirem aqui, não são induction heads copiando o prompt.
    "no_echo": "After a private phone call between their sales managers, two competitors set the same price. Illegal? Just answer Yes or No.",
    # RUN-002: controle de eco + VIÉS DO SLOT Yes/No. Resposta aberta: o
    # verdict É o próprio conceito, então não existe viés de formato Yes.
    # Predição: rank de 'collusion' despenca até ~1 nas camadas finais e a
    # emergência deve ser visível na banda workspace. (Coluna Yes-No: ignorar.)
    "open_concept": "Two competitors secretly meet and then set identical prices. What is the one-word legal term for this practice?",
}

# Condições Yes/No comparáveis entre si (a aberta tem outra escala de resposta)
YESNO_CONDITIONS = ["independent", "collusion", "no_echo"]
BASELINE = "independent"

# Keywords de 1 token para rastrear no J-lens (multi-token é invisível ao lens).
KEYWORDS = ["collusion", "agreement", "conspiracy", "independent", "cartel"]

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    CFG["model_id"], revision=CFG["model_revision"]
)
# Classe HF vem da config: CausalLM (Qwen) ou ImageTextToText (Gemma E4B,
# multimodal — jlens.from_hf autodetecta o decoder em model.language_model).
AutoClass = getattr(transformers, CFG["auto_class"])
hf_model = AutoClass.from_pretrained(
    CFG["model_id"], revision=CFG["model_revision"], dtype=torch.bfloat16
).to(device)
model = jlens.from_hf(hf_model, tokenizer)

# Dois formatos de distribuição de lens no HF Hub:
#   from_pretrained — repo com layout do neuronpedia (metadados + revision)
#   hub_file        — arquivo lens.pt avulso (registro solarkyle/jspace-lenses)
if CFG["lens_loader"] == "from_pretrained":
    lens = jlens.JacobianLens.from_pretrained(
        CFG["lens_repo"], filename=CFG["lens_file"], revision=CFG["lens_revision"]
    )
else:
    from huggingface_hub import hf_hub_download

    lens_path = hf_hub_download(
        CFG["lens_repo"], CFG["lens_file"], revision=CFG["lens_revision"]
    )
    lens = jlens.JacobianLens.load(lens_path)
print(model)
print(lens)

# ---------------------------------------------------------------------------
# Relatório completo vai para um .txt; console fica só com o progresso.
# Truque: redirecionamos sys.stdout para o arquivo (todos os print das
# análises caem lá) e status() escreve na saída original do terminal.
# ---------------------------------------------------------------------------
_console = sys.stdout
_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
REPORT_PATH = f"report_{ACTIVE_MODEL}_{_stamp}.txt"
_report = open(REPORT_PATH, "w")
_report.write(f"# run_local.py — {ACTIVE_MODEL} ({CFG['model_id']}) — {_stamp}\n")
sys.stdout = _report


def status(msg: str) -> None:
    """Linha de progresso no terminal (o stdout está indo para o .txt)."""
    print(msg, file=_console, flush=True)


status(f"relatório completo em: {REPORT_PATH}")

# Camadas vêm do checkpoint do lens, não de um range hardcoded.
LAYERS = lens.source_layers[::2]
N_LAYERS = model.n_layers


def format_prompt(user_text: str) -> str:
    """Chat template + generation prompt (kwargs extras por modelo na config,
    ex.: enable_thinking=False no Qwen; Gemma não tem o switch).

    Com add_generation_prompt=True o texto termina no cabeçalho do turno do
    assistant, então os logits da ÚLTIMA posição (-1) predizem o primeiro
    token da resposta — é ali que lemos o verdict em formação.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        **CFG["chat_template_kwargs"],
    )


def token_variants(word: str) -> dict[int, str]:
    """IDs candidatos para uma palavra: com e sem espaço à frente (BPE difere)."""
    out = {}
    for text in (word, " " + word):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:  # só interessa se for token único
            out[ids[0]] = repr(text)
    return out


def best_variant(word: str, final_logits: torch.Tensor) -> int | None:
    """Entre 'word' e ' word', escolhe o ID que o modelo realmente prefere
    (maior logit na camada final). None se a palavra não é token único."""
    variants = token_variants(word)
    if not variants:
        return None
    return max(variants, key=lambda i: float(final_logits[i]))


SPECIAL_IDS = set(tokenizer.all_special_ids)


def readable_topk(logits: torch.Tensor, k: int = 6) -> list[str]:
    """Top tokens pulando pontuação pura, não-ASCII e tokens especiais do
    template (<|endoftext|>, <|im_end|>, <think>...). Qwen tem muito CJK e
    boilerplate no top-K; o artigo usa máscara/gloss — aqui filtro simples."""
    out = []
    for idx in logits.topk(80).indices:
        if int(idx) in SPECIAL_IDS:
            continue
        tok = tokenizer.decode([idx])
        s = tok.strip()
        if s.startswith("<") and s.endswith(">"):  # <think>, </think>, <|...|>
            continue
        if s and s.isascii() and any(c.isalpha() for c in s):
            out.append(tok)
        if len(out) == k:
            break
    return out


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    """Posição do token no ranking da distribuição (1 = mais provável).

    LEIA ASSIM: o lens produz um logito para CADA um dos ~150k tokens do
    vocabulário. Ordenando todos do maior para o menor, rank = posição do
    nosso token nessa fila. A coluna "top" das tabelas mostra só os ranks
    1-6 (depois do filtro) — por isso uma keyword com rank 700 NUNCA aparece
    lá, e ainda assim está à frente de 99.5% do vocabulário. A tabela de
    ranks é o microscópio para o que o top-6 não mostra.
    """
    return int((logits > logits[token_id]).sum()) + 1


def fmt_rank(rank: int, vocab: int) -> str:
    """Rank + percentil, ex.: '736 (0.5%)' = à frente de 99.5% do vocab."""
    return f"{rank} ({100 * rank / vocab:.1f}%)"


def analyze(label: str, user_text: str) -> dict:
    prompt = format_prompt(user_text)
    print("\n" + "=" * 78)
    print(f"[{label}] {user_text}")
    print("=" * 78)

    # positions=[-1]: última posição do prompt formatado → prediz 1º token da resposta
    jl, model_logits, _ = lens.apply(model, prompt, layers=LAYERS, positions=[-1])
    ll, _, _ = lens.apply(
        model, prompt, layers=LAYERS, positions=[-1], use_jacobian=False
    )
    final = model_logits[0]

    yes_id = best_variant("Yes", final)
    no_id = best_variant("No", final)
    kw_ids = {w: best_variant(w, final) for w in KEYWORDS}
    kw_ids = {w: i for w, i in kw_ids.items() if i is not None}
    print(f"token IDs: Yes={yes_id} No={no_id} | keywords: {kw_ids}\n")

    header = f"{'L':>3} | {'logit-lens top':<38} | {'J-lens top':<38} | {'Yes-No':>7}"
    print(header)
    print("-" * len(header))
    diffs: dict[int, float] = {}
    for layer in LAYERS:
        j = jl[layer][0]
        diff = float(j[yes_id] - j[no_id])
        diffs[layer] = diff
        regime = (
            "*" if N_LAYERS // 3 <= layer < N_LAYERS - 3 else " "
        )  # banda workspace
        print(
            f"{layer:>2}{regime} | {' '.join(readable_topk(ll[layer][0])):<38.38} "
            f"| {' '.join(readable_topk(j)):<38.38} | {diff:>+7.2f}"
        )
    print(
        f"    | {'(modelo, camada final)':<38} "
        f"| {' '.join(readable_topk(final)):<38.38} "
        f"| {float(final[yes_id] - final[no_id]):>+7.2f}"
    )

    # Rank das keywords no J-lens por camada, com percentil para dar escala.
    vocab = final.shape[0]
    print(
        f"\nrank no J-lens — rank (percentil do vocab de {vocab}); "
        f"1 = top absoluto, 1.0% = à frente de 99% dos tokens:"
    )
    print(f"{'L':>3} | " + " | ".join(f"{w:>16}" for w in kw_ids))
    ranks: dict[str, dict[int, int]] = {w: {} for w in kw_ids}
    for layer in LAYERS:
        j = jl[layer][0]
        cells = []
        for w, i in kw_ids.items():
            r = rank_of(j, i)
            ranks[w][layer] = r
            cells.append(f"{fmt_rank(r, vocab):>16}")
        print(f"{layer:>3} | " + " | ".join(cells))

    # Geração real para conferir que o lens está lendo a pergunta certa.
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = hf_model.generate(
        **inputs,
        max_new_tokens=12,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(
        out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    print(f"\nresposta do modelo: {answer.strip()!r}")

    return {
        "diff": diffs,
        "ranks": ranks,
        "final_diff": float(final[yes_id] - final[no_id]),
        "answer": answer.strip(),
        "vocab": vocab,
    }


RESULTS = {}
for label, text in PROMPTS.items():
    RESULTS[label] = analyze(label, text)
    status(f"[{label}] analisado — resposta: {RESULTS[label]['answer']!r}")

# %%%
# ---------------------------------------------------------------------------
# COMPARAÇÃO ENTRE CONDIÇÕES — é aqui que o sinal mora, não nos absolutos
# ---------------------------------------------------------------------------
# (1) O nível absoluto de Yes-No carrega viés de formato ("Yes" é genérico em
#     slot de resposta). O que informa é o DELTA vs a condição baseline
#     'independent', camada a camada.
# (2) Rank absoluto de keyword também não informa sozinho (J-space = 6-10% da
#     variância; lens treinado em wikitext). O que informa é o CONTRASTE do
#     rank da mesma keyword entre condições, na mesma camada.

print("\n" + "=" * 78)
print(f"Δ(Yes-No) vs baseline '{BASELINE}' — positivo = mais inclinado a Yes")
print("=" * 78)
others = [c for c in YESNO_CONDITIONS if c != BASELINE]
print(f"{'L':>3} | " + " | ".join(f"{c:>14}" for c in others))
for layer in LAYERS:
    base = RESULTS[BASELINE]["diff"][layer]
    row = " | ".join(f"{RESULTS[c]['diff'][layer] - base:>+14.2f}" for c in others)
    print(f"{layer:>3} | {row}")
print(
    "    (esperado se hipótese vale: coluna 'collusion' positiva e crescente\n"
    "     na banda workspace; 'no_echo' idem => sinal não é eco lexical)"
)

vocab = RESULTS[BASELINE]["vocab"]
for w in RESULTS[BASELINE]["ranks"]:
    print(f"\nrank de '{w}' no J-lens por condição — rank (percentil):")
    conds = [c for c in PROMPTS if w in RESULTS[c]["ranks"]]
    print(f"{'L':>3} | " + " | ".join(f"{c:>18}" for c in conds))
    for layer in LAYERS:
        row = " | ".join(
            f"{fmt_rank(RESULTS[c]['ranks'][w][layer], vocab):>18}" for c in conds
        )
        print(f"{layer:>3} | {row}")

# %%%
# ---------------------------------------------------------------------------
# VISUALIZAÇÃO jlens — dashboard interativo posições x camadas
# ---------------------------------------------------------------------------
# Gera um HTML autocontido por prompt (mode="embed"). Abra no navegador:
#   open viz_independent.html   (etc.)
# pinned_token_ids = nossas keywords + Yes/No: a página já abre rastreando
# esses tokens em TODAS as posições e camadas — resolve o "não enxergo meus
# tokens no top": lá o rank deles aparece como curva, não só o top-K.
import gzip
import json
from pathlib import Path

from jlens.vis import build_page, compute_slice

# Gloss (tradução de tokens CJK/raros) é específico do vocabulário do modelo.
# Só o Qwen3.5 tem gloss no repo local; sem gloss o dashboard funciona igual.
gloss = None
if CFG["gloss_file"]:
    GLOSS_PATH = Path(__file__).parent / CFG["gloss_file"]
    gloss = {int(k): v for k, v in json.load(gzip.open(GLOSS_PATH)).items()}

pinned = {tid for w in KEYWORDS for tid in token_variants(w)}
pinned |= set(token_variants("Yes")) | set(token_variants("No"))

for label, text in PROMPTS.items():
    slice_data = compute_slice(
        model,
        lens,
        format_prompt(text),
        layer_stride=2,
        mask_display=True,          # esconde tokens ilegíveis do display
        pinned_token_ids=pinned,    # ranks completos das nossas keywords
    )
    extra = {"alt_token": gloss} if gloss else {}
    page, _, _ = build_page(
        slice_data,
        text,
        title=f"J-lens: {label} ({ACTIVE_MODEL})",
        description=text,
        pinned_token_ids=pinned,
        **extra,
    )
    # nome inclui o modelo: runs de modelos diferentes não se sobrescrevem
    out_path = Path(__file__).parent / f"viz_{ACTIVE_MODEL}_{label}.html"
    out_path.write_text(page)
    print(f"visualização salva: {out_path.name}  (abrir com: open {out_path.name})")
    status(f"visualização salva: {out_path.name}")

# %%%
# ---------------------------------------------------------------------------
# SWAP: "manipular o pensamento" (análogo ao spider -> ant do artigo)
# ---------------------------------------------------------------------------
# O artigo troca o vetor lido pelo J-lens em camada média e mostra que a
# resposta final muda ("spider" -> "ant" vira 8 -> 6 pernas). Aqui fazemos a
# versão de primeira ordem no domínio da colusão:
#
#   No prompt "independently" (verdict esperado: No), injetamos no residual
#   da camada L a direção que, segundo o Jacobiano, empurra o conceito
#   'independent' -> 'collusion':
#
#       delta_h = alpha * J_L^T @ (w_collusion - w_independent)
#
#   onde w_* são as linhas (normalizadas) da matriz de unembedding. Como
#   J_L aproxima d(logits finais)/d(h_L), J_L^T @ w é a direção em h_L que
#   mais aumenta o logito daquele token — subir 'collusion' e descer
#   'independent' de uma vez. Se o conceito for causal para o verdict,
#   a resposta deve virar de No para Yes conforme alpha cresce.
#
# alpha=0 é o baseline (sem intervenção). Camada na banda workspace (~2/3).

SWAP_LAYER = min(lens.source_layers, key=lambda l: abs(l - 2 * N_LAYERS // 3))
SWAP_SRC, SWAP_TGT = "independent", "collusion"
SWAP_ALPHAS = [0.0, 4.0, 8.0, 16.0]  # escala é empírica: ajuste se nada mudar


def swap_direction(src: str, tgt: str, layer: int) -> torch.Tensor:
    """Direção de steering em h_layer: alpha * J^T @ (w_tgt - w_src)."""
    W_U = hf_model.get_output_embeddings().weight  # [vocab, d_model]
    src_id = tokenizer.encode(" " + src, add_special_tokens=False)[0]
    tgt_id = tokenizer.encode(" " + tgt, add_special_tokens=False)[0]
    w_src = W_U[src_id].float() / W_U[src_id].float().norm()
    w_tgt = W_U[tgt_id].float() / W_U[tgt_id].float().norm()
    J = lens.jacobians[layer].to(W_U.device)  # [d_model, d_model], float32
    direction = J.T @ (w_tgt - w_src).to(J.device)
    return direction / direction.norm()  # unitária; alpha controla a força


def swap_experiment(user_text: str) -> None:
    prompt = format_prompt(user_text)
    direction = swap_direction(SWAP_SRC, SWAP_TGT, SWAP_LAYER)
    block = model.layers[SWAP_LAYER]  # mesmo módulo usado por lens.apply e generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("\n" + "=" * 78)
    print(f"[swap] '{SWAP_SRC}' -> '{SWAP_TGT}' na camada L{SWAP_LAYER}")
    print(f"prompt: {user_text}")
    print("=" * 78)

    for alpha in SWAP_ALPHAS:
        delta = (alpha * direction).to(dtype=hf_model.dtype)

        def hook(module, args, output):
            # soma delta em TODAS as posições (intervenção simples e didática).
            # conforme a versão do transformers, o bloco decoder retorna o
            # tensor de hidden states direto ou uma tupla com ele em [0]
            if isinstance(output, tuple):
                return (output[0] + delta,) + tuple(output[1:])
            return output + delta

        handle = block.register_forward_hook(hook)
        try:
            # 1. o que o J-lens passa a "ver" na camada seguinte à intervenção
            probe_layers = [l for l in lens.source_layers if l > SWAP_LAYER][:1]
            jl, _, _ = lens.apply(model, prompt, layers=probe_layers, positions=[-1])
            seen = " ".join(readable_topk(jl[probe_layers[0]][0], k=4))
            # 2. a resposta final com o "pensamento" alterado
            out = hf_model.generate(
                **inputs,
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            answer = tokenizer.decode(
                out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            ).strip()
        finally:
            handle.remove()  # SEMPRE remover: hook esquecido contamina tudo depois

        tag = "(baseline)" if alpha == 0 else ""
        print(
            f"alpha={alpha:>5.1f} {tag:<10} J-lens@L{probe_layers[0]}: {seen:<30.30} "
            f"| resposta: {answer!r}"
        )

    print(
        "\nLeitura: se a resposta vira No -> Yes com alpha moderado, o conceito\n"
        "'coordenação' é causalmente ligado ao verdict (não só correlação).\n"
        "Se só vira com alpha enorme (ou a resposta degenera em texto quebrado),\n"
        "a intervenção está gritando por cima do modelo, não revelando estrutura.\n"
        "O artigo reporta ~54-70% de sucesso em swaps: efeito ruidoso é esperado."
    )


status(f"comparações entre condições gravadas em {REPORT_PATH}")
swap_experiment(PROMPTS["independent"])
status(f"swap L{SWAP_LAYER} concluído (resultados no txt)")

print(
    "\nComo ler (detalhes em NOTES.md):\n"
    "  * marca a banda 'workspace' (~L12-L33): só interprete conceitos ali.\n"
    "  - Yes-No > 0 = disposição a 'Yes' (ilegal); a camada onde muda de sinal\n"
    "    localiza onde o verdict comita.\n"
    "  - Assinatura-alvo: keywords com rank baixo em camada média SÓ no prompt\n"
    "    'collusion', antes do crossover Yes-No.\n"
    "  - J-lens != logit-lens em camada média = conteúdo interno genuíno.\n"
)

# Restaura o stdout e fecha o relatório.
sys.stdout = _console
_report.close()
print(f"\npronto. relatório completo: {REPORT_PATH}")
