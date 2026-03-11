# Lab 3 — Transformer Internals

Três exercícios práticos explorando os mecanismos centrais de um Transformer: máscara causal, cross-attention e geração autoregressiva. Nada de framework de alto nível — tudo feito na mão com PyTorch puro para entender o que realmente acontece por baixo.

## O que tem aqui

**task1_causal_mask.py** — Implementa a máscara causal usada no decoder. A ideia é simples: cada token só pode "ver" os tokens anteriores a ele, nunca os futuros. Isso é feito somando `-inf` nas posições futuras antes do softmax, o que as zera efetivamente nos pesos de atenção.

**task2_cross_attention.py** — Implementa o cross-attention entre encoder e decoder. O decoder (em inglês, nesse exemplo) manda as queries, e o encoder (em francês) fornece keys e values. É o mecanismo que permite ao decoder focar nas partes relevantes da sequência de entrada enquanto gera a saída.

**task3_autoregressive_loop.py** — Simula o loop de inferência autoregressiva. O modelo gera um token por vez, adiciona ao histórico e usa tudo isso para gerar o próximo — exatamente como um modelo de tradução ou LLM funciona na prática. A geração para ao encontrar o token `<EOS>` ou ao atingir o limite de steps.

## Como rodar

Precisa ter Python 3 e PyTorch instalados.

```bash
pip install torch
```

Depois é só rodar cada task separadamente:

```bash
python3 task1_causal_mask.py
python3 task2_cross_attention.py
python3 task3_autoregressive_loop.py
```

Ou tudo de uma vez:

```bash
python3 task1_causal_mask.py && python3 task2_cross_attention.py && python3 task3_autoregressive_loop.py
```

## Observação

Os pesos são inicializados aleatoriamente (`torch.randn`), então os valores numéricos vão mudar a cada execução — o que importa é o formato das saídas e o comportamento do mecanismo, não os números em si.
