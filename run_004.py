# RUN-004 — sair do n=1: paráfrases, controle negativo, doador forte.
# Desenho e predições pré-registradas: EXPERIMENTS.md, seção RUN-004.
#
# COMO RODAR (mesmo ambiente; modelo em cache):
#     uv run python run_004.py
#
# Braço A: 6 paráfrases × {independent, collusion, no_echo} — média ± desvio
#          de Δ(Yes−No) por camada e ranks de 'independent'.
# Braço B: 3 controles negativos (contato inócuo + preços iguais).
# Braço C: patch real com doador de margem folgada (cartel explícito).
#
# Output completo → report_<modelo>_run004_<timestamp>.txt; console = progresso.

# %%%
import datetime
import os
import statistics
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

_console = sys.stdout
_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
REPORT_PATH = f"report_{ACTIVE_MODEL}_run004_{_stamp}.txt"
_report = open(REPORT_PATH, "w")
_report.write(f"# run_004.py — {ACTIVE_MODEL} ({CFG['model_id']}) — {_stamp}\n")
sys.stdout = _report


def status(msg: str) -> None:
    print(msg, file=_console, flush=True)


status(f"relatório completo em: {REPORT_PATH}")

# %%%
# ---------------------------------------------------------------------------
# BATERIA DE PROMPTS (pré-registrada)
# ---------------------------------------------------------------------------
# no_echo: proibidas as palavras-gatilho agree/collude/cartel/conspire/
# coordinate/combine/pact/fix (e derivados). Só descrição de contato + preço.
SUFFIX = " Illegal? Just answer Yes or No."

BATTERY = {
    "independent": [
        "Two competitors independently choose the same price.",
        "Without ever communicating, two rival firms end up charging the same price.",
        "Two rival shops happen to set identical prices, each on its own.",
        "Each of two competing companies sets its price alone, and the prices turn out equal.",
        "Watching the market separately, two competing sellers arrive at the same price.",
        "Reacting to the same rise in costs, two competitors price their product identically.",
    ],
    "collusion": [
        "Two competitors combine to choose the same price.",
        "Two rival firms agree to charge the same price.",
        "Two competing companies make a pact to keep their prices identical.",
        "Two competitors form a cartel and set identical prices.",
        "In a secret deal, two rival sellers coordinate their pricing.",
        "Two competing shops conspire to keep prices equal.",
    ],
    "no_echo": [
        "After a private phone call between their sales managers, two competitors set the same price.",
        "Following a quiet dinner between their CEOs, two rival firms raise prices together the next morning.",
        "After exchanging late-night emails, two competing companies post the same new price list.",
        "Two competitors meet in a hotel room; a week later both charge identical prices.",
        "After their managers talk privately at a trade fair, two rival sellers move to the same price.",
        "Once their owners start texting every day, two competing shops always match prices.",
    ],
    # Braço B — controle negativo: contato explícito, conteúdo inócuo.
    "neg_control": [
        "After chatting about the weather, two competitors happen to charge the same price.",
        "Their managers played golf and talked only about football; later both firms set the same price.",
        "After meeting at a charity event to plan a fundraiser, two rival shops end up with identical prices.",
    ],
}

KEYWORD = "independent"  # keyword de dissociação (achado central da RUN-002)
# Bandas de camada vêm do config do modelo (recalibradas por profundidade);
# fallback = valores originais do Qwen (36 camadas).
_BANDS = CFG.get("bands", {})
BAND = _BANDS.get("agg_band", [16, 18, 20, 22, 24, 26, 28])  # agregados
RANK_BAND = _BANDS.get("rank_band", [18, 20, 22, 24, 26])


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


LAYERS = lens.source_layers[::2]


def probe(user_text: str) -> dict:
    """J-lens na última posição: Δ(Yes−No) por camada, rank da keyword,
    margem final e resposta gerada."""
    prompt = format_prompt(user_text)
    jl, model_logits, _ = lens.apply(model, prompt, layers=LAYERS, positions=[-1])
    final = model_logits[0]
    yes_id = best_variant("Yes", final)
    no_id = best_variant("No", final)
    kw_id = best_variant(KEYWORD, final)

    diffs, kranks = {}, {}
    for layer in LAYERS:
        j = jl[layer][0]
        diffs[layer] = float(j[yes_id] - j[no_id])
        kranks[layer] = int((j > j[kw_id]).sum()) + 1

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = hf_model.generate(
            **inputs, max_new_tokens=6, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(
        out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()
    return {
        "diff": diffs,
        "kw_rank": kranks,
        "final_margin": float(final[yes_id] - final[no_id]),
        "answer": answer,
    }


# %%%
# ---------------------------------------------------------------------------
# BRAÇOS A + B — bateria
# ---------------------------------------------------------------------------
RESULTS: dict[str, list[dict]] = {}
for cond, texts in BATTERY.items():
    RESULTS[cond] = []
    print("\n" + "=" * 78)
    print(f"[{cond}] {len(texts)} paráfrases")
    print("=" * 78)
    for i, text in enumerate(texts):
        r = probe(text + SUFFIX)
        RESULTS[cond].append(r)
        print(f"  {i}: {r['answer']!r:<8} margem {r['final_margin']:+6.2f} | {text}")
        status(f"[{cond}] {i + 1}/{len(texts)}: {r['answer']!r}")

# Agregados: contagem de respostas + Δ(Yes−No) médio±dp vs média independent
print("\n" + "=" * 78)
print("AGREGADOS — respostas por condição")
print("=" * 78)
for cond, rs in RESULTS.items():
    answers = [r["answer"].split()[0] if r["answer"] else "?" for r in rs]
    yes = sum(a.lower().startswith("yes") for a in answers)
    no = sum(a.lower().startswith("no") for a in answers)
    print(f"{cond:>12}: Yes={yes} No={no} outros={len(rs) - yes - no} | {answers}")

base_mean = {
    layer: statistics.mean(r["diff"][layer] for r in RESULTS["independent"])
    for layer in LAYERS
}

print("\nΔ(Yes−No) vs média do independent — média ± desvio por condição:")
conds = [c for c in BATTERY if c != "independent"]
print(f"{'L':>3} | " + " | ".join(f"{c:>22}" for c in conds))
for layer in LAYERS:
    cells = []
    for c in conds:
        vals = [r["diff"][layer] - base_mean[layer] for r in RESULTS[c]]
        m = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        cells.append(f"{m:+7.2f} ± {sd:5.2f}")
    print(f"{layer:>3} | " + " | ".join(f"{c:>22}" for c in cells))
print(
    "  (predição 2: collusion e no_echo com média−desvio > 0 na banda L16–L28;\n"
    "   predição 4: neg_control < metade do no_echo)"
)

print(f"\nrank de '{KEYWORD}' na banda L18–L26 — mediana [min–max] por condição:")
for cond, rs in RESULTS.items():
    pooled = [r["kw_rank"][layer] for r in rs for layer in RANK_BAND]
    med = statistics.median(pooled)
    print(
        f"{cond:>12}: mediana {med:>8.0f} "
        f"[{min(pooled):>6} – {max(pooled):>7}] (n={len(pooled)})"
    )
print("  (predição 3: independent < 20k; collusion/no_echo > 40k)")
status("braços A+B agregados")

# %%%
# ---------------------------------------------------------------------------
# BRAÇO C — patch real com doador forte
# ---------------------------------------------------------------------------
STRONG_DONOR = (
    "Two competitors form a secret cartel agreement to fix their prices"
    " at the same level." + SUFFIX
)
TARGET = BATTERY["independent"][0] + SUFFIX
PATCH_LAYERS = _BANDS.get("patch004", [18, 20, 22, 24, 26])

TARGET_INPUTS = tokenizer(format_prompt(TARGET), return_tensors="pt").to(device)
DONOR_INPUTS = tokenizer(format_prompt(STRONG_DONOR), return_tensors="pt").to(device)

with torch.no_grad():
    base_logits = hf_model(**TARGET_INPUTS).logits[0, -1].float()
YES_ID = best_variant("Yes", base_logits)
NO_ID = best_variant("No", base_logits)
BASE_MARGIN = float(base_logits[YES_ID] - base_logits[NO_ID])

donor_probe = probe(STRONG_DONOR)
print("\n" + "=" * 78)
print("BRAÇO C — patch real com doador forte")
print(f"doador: {STRONG_DONOR}")
print(
    f"doador sozinho: resposta {donor_probe['answer']!r}, "
    f"margem {donor_probe['final_margin']:+.2f}  (predição 5: > +2)"
)
print(f"alvo:   {TARGET}  (margem baseline {BASE_MARGIN:+.2f})")
print("=" * 78)
status(
    f"doador forte: {donor_probe['answer']!r} margem {donor_probe['final_margin']:+.2f}"
)


def block_output(output):
    return output[0] if isinstance(output, tuple) else output


def with_hidden(output, h):
    if isinstance(output, tuple):
        return (h,) + tuple(output[1:])
    return h


def capture_donor_states() -> dict[int, torch.Tensor]:
    captured, handles = {}, []
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

print(f"{'L':>3} | {'resposta':<10} | {'margem Yes−No':>14} | {'Δ vs baseline':>14}")
print("-" * 52)
flips = 0
for layer in PATCH_LAYERS:
    handle = model.layers[layer].register_forward_hook(
        make_patch_hook(donor_states[layer])
    )
    try:
        with torch.no_grad():
            out = hf_model.generate(
                **TARGET_INPUTS, max_new_tokens=6, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            answer = tokenizer.decode(
                out[0][TARGET_INPUTS.input_ids.shape[1] :], skip_special_tokens=True
            ).strip()
            logits = hf_model(**TARGET_INPUTS).logits[0, -1].float()
    finally:
        handle.remove()
    margin = float(logits[YES_ID] - logits[NO_ID])
    is_flip = answer[:3].lower() == "yes"
    flips += is_flip
    print(
        f"{layer:>3} | {answer[:10]!r:<10} | {margin:>+14.2f} "
        f"| {margin - BASE_MARGIN:>+14.2f}{'  «FLIP»' if is_flip else ''}"
    )
    status(f"braço C: L{layer} — {answer!r} (margem {margin:+.2f})")

print(
    f"\nflips: {flips}/{len(PATCH_LAYERS)} "
    "(predição 5: ≥3/5 com margem final > +1 = flip robusto)"
)

sys.stdout = _console
_report.close()
print(f"\npronto. relatório completo: {REPORT_PATH}")
