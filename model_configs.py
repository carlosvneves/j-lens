# Configuração de modelos + lenses para run_local.py.
#
# Escolha do modelo: variável de ambiente JSPACE_MODEL ou DEFAULT_MODEL abaixo.
#   JSPACE_MODEL=gemma-4-e4b-it uv run python run_local.py
#
# Cada entrada declara onde vive o modelo, qual classe HF o carrega, onde vive
# o lens correspondente e como carregá-lo. LENS E MODELO SÃO UM PAR: Jacobianos
# são específicos por modelo/camada — nunca misturar lens de um com pesos de
# outro (nem base vs -it do mesmo modelo).
#
# Fonte primária de lens: neuronpedia/jacobian-lens (branch main, ~30 modelos,
# fits n=1000 wikitext p/ os que usamos). solarkyle/jspace-lenses só quando o
# neuronpedia não cobre (ex.: gemma-4-E4B-it — o neuronpedia tem só o E4B base).
#
# Nota histórica (2026-07-18): Qwen/Qwen3.5-4B deu 404 no Hub por algumas horas
# e voltou. Se sumir de novo, alternativa local: gemma-4-e4b-it.

DEFAULT_MODEL = "qwen3.5-4b"

MODEL_CONFIGS = {
    # Config da RUN-001/RUN-002 (local, M5/24GB). Lens n=1000 wikitext,
    # agora na branch main do neuronpedia (a revision 'qwen-n1000' virou
    # desnecessária).
    "qwen3.5-4b": {
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": None,
        "auto_class": "AutoModelForCausalLM",
        "lens_loader": "from_pretrained",  # jlens.JacobianLens.from_pretrained
        "lens_repo": "neuronpedia/jacobian-lens",
        "lens_file": "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
        "lens_revision": None,
        # Qwen é reasoning: desligar thinking p/ verdict = 1º token gerado.
        "chat_template_kwargs": {"enable_thinking": False},
        "gloss_file": "jacobian-lens/assets/qwen_gloss.json.gz",
        "approx_download_gb": 9,
        # Bandas de camada (36 camadas). Originais das RUN-003/004 — manter
        # p/ comparabilidade com resultados já registrados.
        "bands": {
            "steer": [16, 18, 20, 22, 24, 26],          # swap003 braço A
            "patch003": [14, 16, 18, 20, 22, 24, 26, 28],  # swap003 braço B
            "agg_band": [16, 18, 20, 22, 24, 26, 28],   # run_004 agregados
            "rank_band": [18, 20, 22, 24, 26],          # run_004 rank keyword
            "patch004": [18, 20, 22, 24, 26],           # run_004 braço C
        },
    },
    # Alternativa local (mesmo porte, outra família — teste de robustez ou
    # fallback se o Qwen sumir do Hub de novo). Lens do solarkyle (fit n=100,
    # bf16) porque o neuronpedia só tem o E4B BASE, e o modelo aqui é o -it.
    # Gemma E4B é multimodal → AutoModelForImageTextToText; jlens.from_hf
    # autodetecta o layout (model.language_model.layers). Exige aceitar a
    # licença Gemma no HF + token de leitura.
    "gemma-4-e4b-it": {
        "model_id": "google/gemma-4-E4B-it",
        "model_revision": None,
        "auto_class": "AutoModelForImageTextToText",
        "lens_loader": "hub_file",  # hf_hub_download + JacobianLens.load
        "lens_repo": "solarkyle/jspace-lenses",
        "lens_file": "gemma-4-e4b-it/lens.pt",
        "lens_revision": None,
        # Template do Gemma não tem switch de thinking — sem kwargs extras.
        "chat_template_kwargs": {},
        # Gloss é específico do vocabulário; o do repo local é Qwen.
        "gloss_file": None,
        "approx_download_gb": 8,
        # Bandas recalibradas p/ 42 camadas (RUN-005 usou as do Qwen — confound
        # registrado no EXPERIMENTS.md). Base: (a) proporcional 42/36 sobre a
        # banda Qwen 16–26 → 19–30; (b) evidência da RUN-005: patch só mexe a
        # margem a partir de L24, doador forte flipa L24/L26, Δ ainda subindo
        # em L28; Δ collusion explode em L30+. Mantém L22 no steer (flip
        # observado lá).
        "bands": {
            "steer": [18, 20, 22, 24, 26, 28, 30],
            "patch003": [20, 22, 24, 26, 28, 30, 32, 34],
            "agg_band": [20, 22, 24, 26, 28, 30, 32],
            "rank_band": [22, 24, 26, 28, 30],
            "patch004": [22, 24, 26, 28, 30],
        },
    },
    "gemma-4-e4b": {
        "model_id": "google/gemma-4-e4b",
        "model_revision": None,
        "auto_class": "AutoModelForImageTextToText",
        "lens_loader": "hub_file",
        "lens_repo": "neuronpedia/jacobian-lens",
        "lens_file": "gemma-4-e4b/jlens/Salesforce-wikitext/gemma-4-E4B_jacobian_lens.pt",
        "lens_revision": None,
        "chat_template_kwargs": {},
        "gloss_file": None,
        "approx_download_gb": 8,
    },
    # Replicação em escala (Colab A100, não local — ~55 GB bf16). Lens n=1000
    # da mesma fonte primária.
    "qwen3.6-27b": {
        "model_id": "Qwen/Qwen3.6-27B",
        "model_revision": None,
        "auto_class": "AutoModelForCausalLM",
        "lens_loader": "from_pretrained",
        "lens_repo": "neuronpedia/jacobian-lens",
        "lens_file": "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
        "lens_revision": None,
        "chat_template_kwargs": {"enable_thinking": False},
        "gloss_file": None,  # gloss local é do vocab Qwen3.5; validar antes de reusar
        "approx_download_gb": 55,
    },
}
