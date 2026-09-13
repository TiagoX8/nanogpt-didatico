"""
prepare.py — Tokenização em nível de caractere (char-level).

Um modelo de linguagem não entende letras: ele só trabalha com NÚMEROS.
O primeiro passo, portanto, é transformar o texto em uma sequência de inteiros.
Esse processo chama-se TOKENIZAÇÃO.

Modelos grandes (GPT-4, Llama...) usam tokenizadores de "subpalavras" (BPE),
em que um token pode ser um pedaço de palavra ("apren", "dizado"). Aqui, para
fins didáticos, usamos a versão mais simples possível: CADA CARACTERE É UM TOKEN.

    "ola" -> ['o', 'l', 'a'] -> [58, 55, 44]

Vantagens: vocabulário minúsculo (~100 símbolos), zero dependências, fácil de
entender. Desvantagem: sequências ficam longas e o modelo precisa "aprender a
soletrar" antes de aprender gramática.

O que este script faz:
  1. Lê o texto bruto (data/dom_casmurro.txt — Machado de Assis, domínio público).
  2. Descobre o vocabulário: o conjunto de caracteres únicos do texto.
  3. Cria os mapeamentos  char -> id (stoi)  e  id -> char (itos).
  4. Divide o texto em treino (90%) e validação (10%).
  5. Codifica tudo em inteiros e salva em disco (train.bin, val.bin, meta.pkl).

Uso:
    python data/prepare.py
"""

import os
import pickle

import numpy as np

# Pasta onde este arquivo está (data/). Usamos isso para que o script funcione
# independentemente de onde o usuário o executa.
PASTA = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_TEXTO = os.path.join(PASTA, "dom_casmurro.txt")

# ---------------------------------------------------------------------------
# 1. Ler o texto bruto
# ---------------------------------------------------------------------------
with open(ARQUIVO_TEXTO, "r", encoding="utf-8") as f:
    texto = f.read()

print(f"Tamanho do texto: {len(texto):,} caracteres")

# ---------------------------------------------------------------------------
# 2. Construir o vocabulário
# ---------------------------------------------------------------------------
# set() remove duplicatas; sorted() garante uma ordem fixa e reprodutível.
chars = sorted(set(texto))
vocab_size = len(chars)
print(f"Vocabulário ({vocab_size} símbolos): {''.join(chars)!r}")

# ---------------------------------------------------------------------------
# 3. Mapeamentos caractere <-> inteiro
# ---------------------------------------------------------------------------
# stoi = "string to int", itos = "int to string" (nomes herdados do nanoGPT).
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s: str) -> list[int]:
    """Texto -> lista de inteiros."""
    return [stoi[c] for c in s]


def decode(ids: list[int]) -> str:
    """Lista de inteiros -> texto."""
    return "".join(itos[i] for i in ids)


# Sanidade: codificar e decodificar deve devolver o texto original.
assert decode(encode("Capitú")) == "Capitú"

# ---------------------------------------------------------------------------
# 4. Separar treino e validação
# ---------------------------------------------------------------------------
# O conjunto de VALIDAÇÃO nunca é usado para ajustar os pesos. Ele serve para
# medir se o modelo está de fato aprendendo padrões gerais da língua ou apenas
# memorizando o texto de treino (overfitting).
n = len(texto)
texto_treino = texto[: int(n * 0.9)]
texto_val = texto[int(n * 0.9):]

# ---------------------------------------------------------------------------
# 5. Codificar e salvar
# ---------------------------------------------------------------------------
ids_treino = encode(texto_treino)
ids_val = encode(texto_val)
print(f"Treino: {len(ids_treino):,} tokens | Validação: {len(ids_val):,} tokens")

# uint16 basta porque vocab_size < 65.536. Salvamos como binário "cru" para
# que train.py consiga carregar rapidamente com np.memmap / np.fromfile.
np.array(ids_treino, dtype=np.uint16).tofile(os.path.join(PASTA, "train.bin"))
np.array(ids_val, dtype=np.uint16).tofile(os.path.join(PASTA, "val.bin"))

# meta.pkl guarda o vocabulário para que generate.py saiba converter de volta
# os inteiros gerados em texto legível.
meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
with open(os.path.join(PASTA, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("Arquivos salvos em data/: train.bin, val.bin, meta.pkl")
