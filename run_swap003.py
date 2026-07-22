# RUN-003 — diagnóstico causal do swap (ver EXPERIMENTS.md, seção RUN-003).
#
# COMO RODAR (mesmo ambiente do run_local.py; modelo já em cache):
#     uv run python run_swap003.py
#   Outro modelo do registro:
#     JSPACE_MODEL=gemma-4-e4b-it uv run python run_swap003.py
#
# O QUE FAZ
# ---------
# A RUN-002 mostrou que dá para ESCREVER 'collusion' no subespaço legível
# (J-lens) sem que o verdict mude. Este script isola o porquê com dois braços:
#
#   Braço A — steering de 1ª ordem: delta_h = α·J̄ᵀ(w_collusion − w_independent)
#       somado no residual, numa grade camada × α × escopo posicional
#       (todas as posições do prompt vs só a última).
#   Braço B — patching real: SUBSTITUI o residual da última posição do prompt
#       'independent' pelo residual da última posição do prompt 'collusion',
#       camada a camada. Sem aproximação de 1ª ordem: é o teste decisivo de
#       dependência causal do verdict em relação àquele estado.
#
# Métricas por célula: resposta gerada (greedy) + margem logit(Yes)−logit(No)
# na camada final. A margem mostra movimento mesmo quando não há flip.
#
# Intervenções só no PREFILL (o forward do prompt inteiro). Nos passos de
# decode (seq_len==1, com KV cache) o hook não faz nada — o efeito da
# intervenção propaga pelo cache. Output completo → report_*.txt.

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

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    CFG["model_id"], revision=CFG["model_revision"]
)
AutoClass = getattr(transformers, CFG["auto_class"])
hf_model = AutoClass.from_pretrained(
    CFG["model_id"], revision=CFG["model_revision"], dtype=torch.bfloat16
).to(device)
model = jlens.from_hf(hf_model, tokenizer)

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

# --- console mínimo + relatório em txt (mesmo esquema do run_local.py) -----
_console = sys.stdout
_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
REPORT_PATH = f"report_{ACTIVE_MODEL}_swap003_{_stamp}.txt"
_report = open(REPORT_PATH, "w")
_report.write(f"# run_swap003.py — {ACTIVE_MODEL} ({CFG['model_id']}) — {_stamp}\n")
sys.stdout = _report


def status(msg: str) -> None:
    print(msg, file=_console, flush=True)


status(f"relatório completo em: {REPORT_PATH}")

# --- desenho experimental (pré-registrado no EXPERIMENTS.md) ----------------
SRC_PROMPT = "Two competitors independently choose the same price. Illegal? Just answer Yes or No."
DONOR_PROMPT = "Two competitors combine to choose the same price. Illegal? Just answer Yes or No."
SWAP_SRC, SWAP_TGT = "independent", "collusion"

# Bandas de camada vêm do config do modelo (recalibradas por profundidade);
# fallback = valores originais do Qwen (36 camadas).
_BANDS = CFG.get("bands", {})
STEER_LAYERS = _BANDS.get("steer", [16, 18, 20, 22, 24, 26])
STEER_ALPHAS = [4.0, 8.0, 16.0]
PATCH_LAYERS = _BANDS.get("patch003", [14, 16, 18, 20, 22, 24, 26, 28])


def format_prompt(user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        **CFG["chat_template_kwargs"],
    )


def best_variant(word: str, final_logits: torch.Tensor) -> int:
    ids = [
        tok[0]
        for text in (word, " " + word)
        if len(tok := tokenizer.encode(text, add_special_tokens=False)) == 1
    ]
    return max(ids, key=lambda i: float(final_logits[i]))


SRC_INPUTS = tokenizer(format_prompt(SRC_PROMPT), return_tensors="pt").to(device)
DONOR_INPUTS = tokenizer(format_prompt(DONOR_PROMPT), return_tensors="pt").to(device)

with torch.no_grad():
    base_logits = hf_model(**SRC_INPUTS).logits[0, -1].float()
YES_ID = best_variant("Yes", base_logits)
NO_ID = best_variant("No", base_logits)
print(f"token IDs: Yes={YES_ID} No={NO_ID}")


def block_output(output):
    """Bloco decoder retorna tensor ou tupla com hidden states em [0]."""
    return output[0] if isinstance(output, tuple) else output


def with_hidden(output, h):
    if isinstance(output, tuple):
        return (h,) + tuple(output[1:])
    return h


def measure(hook_layer: int | None, hook_fn) -> tuple[str, float]:
    """Resposta greedy + margem final Yes−No, com hook opcional na camada.

    O hook fica ativo no generate (prefill + decode) e num forward extra para
    a margem; cabe ao hook_fn ignorar os passos de decode (seq_len==1).
    """
    handle = None
    if hook_layer is not None:
        handle = model.layers[hook_layer].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            out = hf_model.generate(
                **SRC_INPUTS,
                max_new_tokens=6,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            answer = tokenizer.decode(
                out[0][SRC_INPUTS.input_ids.shape[1] :], skip_special_tokens=True
            ).strip()
            logits = hf_model(**SRC_INPUTS).logits[0, -1].float()
    finally:
        if handle is not None:
            handle.remove()  # SEMPRE remover: hook esquecido contamina tudo
    return answer, float(logits[YES_ID] - logits[NO_ID])


# %%%
# ---------------------------------------------------------------------------
# BASELINE (sem intervenção)
# ---------------------------------------------------------------------------
base_answer, base_margin = measure(None, None)
print("\n" + "=" * 78)
print(f"[baseline] {SRC_PROMPT}")
print(f"resposta: {base_answer!r} | margem Yes−No: {base_margin:+.2f}")
print("=" * 78)
status(f"baseline: {base_answer!r} (margem {base_margin:+.2f})")

# %%%
# ---------------------------------------------------------------------------
# BRAÇO A — steering de 1ª ordem: camada × α × escopo posicional
# ---------------------------------------------------------------------------
W_U = hf_model.get_output_embeddings().weight  # [vocab, d_model]


def steer_direction(layer: int) -> torch.Tensor:
    """Direção unitária em h_layer: J̄ᵀ(w_tgt − w_src), como na RUN-002."""
    src_id = tokenizer.encode(" " + SWAP_SRC, add_special_tokens=False)[0]
    tgt_id = tokenizer.encode(" " + SWAP_TGT, add_special_tokens=False)[0]
    w_src = W_U[src_id].float() / W_U[src_id].float().norm()
    w_tgt = W_U[tgt_id].float() / W_U[tgt_id].float().norm()
    J = lens.jacobians[layer].to(W_U.device)
    direction = J.T @ (w_tgt - w_src).to(J.device)
    return direction / direction.norm()


def make_steer_hook(delta: torch.Tensor, last_only: bool):
    def hook(module, args, output):
        h = block_output(output)
        if h.shape[1] == 1:  # passo de decode: não intervir
            return output
        if last_only:
            h = h.clone()
            h[:, -1, :] += delta
        else:
            h = h + delta
        return with_hidden(output, h)

    return hook


print("\n" + "=" * 78)
print(f"BRAÇO A — steering '{SWAP_SRC}' → '{SWAP_TGT}' (1ª ordem, prefill)")
print(f"baseline: resposta {base_answer!r}, margem {base_margin:+.2f}")
print("=" * 78)
header = f"{'L':>3} | {'escopo':<7} | " + " | ".join(
    f"{'α=' + str(int(a)):>16}" for a in STEER_ALPHAS
)
print(header)
print("-" * len(header))
for layer in STEER_LAYERS:
    direction = steer_direction(layer)
    for last_only in (False, True):
        scope = "última" if last_only else "todas"
        cells = []
        for alpha in STEER_ALPHAS:
            delta = (alpha * direction).to(dtype=hf_model.dtype, device=device)
            answer, margin = measure(layer, make_steer_hook(delta, last_only))
            flip = "«FLIP»" if answer[:3].lower() == "yes" else ""
            cells.append(f"{answer[:6]!r} {margin:+.1f}{flip:>6}")
        print(f"{layer:>3} | {scope:<7} | " + " | ".join(f"{c:>16}" for c in cells))
    status(f"braço A: L{layer} concluída")
print("\n(margem baseline era %+.2f; flip = resposta virou Yes)" % base_margin)

# %%%
# ---------------------------------------------------------------------------
# BRAÇO B — patching real: residual da última posição, collusion → independent
# ---------------------------------------------------------------------------
# 1. captura h_L[-1] do prompt DOADOR (collusion) em todas as camadas de
#    interesse numa única passada;
# 2. no prompt alvo (independent), SUBSTITUI h_L[-1] no prefill.


def capture_donor_states() -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in PATCH_LAYERS:

        def hook(module, args, output, layer=layer):
            captured[layer] = block_output(output)[0, -1, :].detach().clone()

        handles.append(model.layers[layer].register_forward_hook(hook))
    try:
        with torch.no_grad():
            hf_model(**DONOR_INPUTS)
    finally:
        for h in handles:
            h.remove()
    return captured


def make_patch_hook(h_donor: torch.Tensor):
    def hook(module, args, output):
        h = block_output(output)
        if h.shape[1] == 1:  # decode: não intervir
            return output
        h = h.clone()
        h[:, -1, :] = h_donor.to(h.dtype)
        return with_hidden(output, h)

    return hook


donor_states = capture_donor_states()
status("braço B: estados do doador capturados")

print("\n" + "=" * 78)
print("BRAÇO B — patching real da última posição (substituição, não soma)")
print(f"doador: {DONOR_PROMPT}")
print(f"alvo:   {SRC_PROMPT}")
print("=" * 78)
print(f"{'L':>3} | {'resposta':<10} | {'margem Yes−No':>14} | {'Δ vs baseline':>14}")
print("-" * 52)
for layer in PATCH_LAYERS:
    answer, margin = measure(layer, make_patch_hook(donor_states[layer]))
    flip = "  «FLIP»" if answer[:3].lower() == "yes" else ""
    print(
        f"{layer:>3} | {answer[:10]!r:<10} | {margin:>+14.2f} "
        f"| {margin - base_margin:>+14.2f}{flip}"
    )
    status(f"braço B: L{layer} — {answer!r} (margem {margin:+.2f})")

print(
    "\nComo ler:\n"
    "  - Braço B com FLIP em alguma camada L16–L26 => o verdict depende\n"
    "    causalmente do residual da última posição nessa banda (predição 1).\n"
    "  - Braço A 'última' > 'todas' => conflito posicional explicava parte\n"
    "    da falha da RUN-002 (predição 2).\n"
    "  - Margem imóvel sob steering mesmo com α=16 => direção J̄ᵀ é ortogonal\n"
    "    ao circuito do verdict: o lens LÊ um subespaço que não é o canal\n"
    "    causal (predição 3 — leitura ≠ escrita).\n"
)

sys.stdout = _console
_report.close()
print(f"\npronto. relatório completo: {REPORT_PATH}")
