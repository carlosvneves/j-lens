# j-space — Jacobian Lens / Global Workspace aplicado à colusão algorítmica

Reprodução do artigo Anthropic (transformer-circuits.pub/2026/workspace) para
pesquisa de colusão algorítmica. Conceitos e plano: `NOTES.md`. Diário de
laboratório (condições → resultados → modificações): `EXPERIMENTS.md` —
registrar toda run e toda mudança de desenho experimental lá. Run local
didático: `run_local.py`. Export original do Colab (com bugs conhecidos,
manter como referência): `j_space_analysis.py`.

## Modo de trabalho (dispatch de modelos)

- **Fable-5 [frontier model]**: elabora plano de ataque, agent dispatch e
  red team (red team quando o usuário pedir).
- **Opus / Fable-5**: explicam pontos difíceis (conceitos, matemática,
  interpretação de resultados).
- **Sonnet**: tarefas operacionais de alta/média complexidade
  (code review, refactor, implementação guiada).
- **Haiku**: tarefas operacionais de baixa complexidade
  (buscas, formatação, tarefas mecânicas).

## Regras

- Passo-a-passo: uma etapa por vez, validada pelo usuário antes de avançar.
- Não executar scripts sem pedido explícito — entregar código + instruções
  de execução em comentário/README.
