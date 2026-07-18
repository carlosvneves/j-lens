# Configuração de modelos + lenses para run_local.py.
#
# Escolha do modelo: variável de ambiente JSPACE_MODEL ou DEFAULT_MODEL abaixo.
#   JSPACE_MODEL=qwen3.5-4b uv run python run_local.py
#
# Cada entrada declara onde vive o modelo, qual classe HF o carrega, onde vive
# o lens correspondente e como carregá-lo. LENS E MODELO SÃO UM PAR: Jacobianos
# são específicos por modelo/camada — nunca misturar lens de um com pesos de
# outro.

DEFAULT_MODEL = "gemma-4-e4b-it"

MODEL_CONFIGS = {
    # Substituto local atual (M5/24GB). Lens do registro solarkyle/jspace-lenses
    # (fit: 100 prompts WikiText-103, bf16). Gemma E4B é multimodal → classe
    # AutoModelForImageTextToText; jlens.from_hf autodetecta o layout
    # (model.language_model.layers).
    "gemma-4-e4b-it": {
        "model_id": "google/gemma-4-E4B-it",
        "model_revision": None,  # TODO: pinar (repo jspace pina revisões; conferir docs)
        "auto_class": "AutoModelForImageTextToText",
        "lens_loader": "hub_file",  # hf_hub_download + JacobianLens.load
        "lens_repo": "solarkyle/jspace-lenses",
        "lens_file": "gemma-4-e4b-it/lens.pt",
        "lens_revision": None,
        # Template do Gemma não tem switch de thinking — sem kwargs extras.
        "chat_template_kwargs": {},
        # Gloss é específico do vocabulário; o do repo local é Qwen. Sem gloss
        # para Gemma por ora (dashboard funciona, só sem tradução de tokens CJK).
        "gloss_file": None,
        "approx_download_gb": 8,
    },
    # Config original da RUN-001. INDISPONÍVEL: Qwen/Qwen3.5-4B foi removido do
    # HF Hub (404 em 2026-07-18). Mantida para reprodutibilidade da RUN-001 e
    # caso reapareça/espelho.
    "qwen3.5-4b": {
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": None,
        "auto_class": "AutoModelForCausalLM",
        "lens_loader": "from_pretrained",  # jlens.JacobianLens.from_pretrained
        "lens_repo": "neuronpedia/jacobian-lens",
        "lens_file": "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
        "lens_revision": "qwen-n1000",
        # Qwen é reasoning: desligar thinking p/ verdict = 1º token gerado.
        "chat_template_kwargs": {"enable_thinking": False},
        "gloss_file": "jacobian-lens/assets/qwen_gloss.json.gz",
        "approx_download_gb": 9,
    },
    # Replicação em escala (Colab A100, não local — ~55 GB bf16).
    "qwen3.6-27b": {
        "model_id": "Qwen/Qwen3.6-27B",
        "model_revision": None,
        "auto_class": "AutoModelForCausalLM",
        "lens_loader": "hub_file",
        "lens_repo": "solarkyle/jspace-lenses",
        "lens_file": "qwen3.6-27b/lens.pt",
        "lens_revision": None,
        "chat_template_kwargs": {"enable_thinking": False},
        "gloss_file": None,  # gloss local é do vocab Qwen3.5; validar antes de reusar
        "approx_download_gb": 55,
    },
}
