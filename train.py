"""
train.py — O loop de treino.

Treinar um modelo de linguagem é surpreendentemente simples de descrever:

    repita muitas vezes:
        1. pegue um pedaço aleatório do texto (x) e o mesmo pedaço deslocado
           em 1 caractere (y) — y é "a resposta certa" para cada posição de x;
        2. passe x pelo modelo e obtenha a previsão para o próximo token;
        3. meça o erro (loss) entre a previsão e y com cross-entropy;
        4. calcule, via backpropagation, quanto cada peso contribuiu para o erro;
        5. dê um pequeno passo nos pesos na direção que reduz o erro (otimizador).

Exemplo com block_size = 8 e o texto "Capitú, Capitú":
    x = "Capitú, "   ->  y = "apitú, C"
    Na posição 0, dado "C",         o alvo é "a"
    Na posição 1, dado "Ca",        o alvo é "p"
    ...
    Na posição 7, dado "Capitú, ",  o alvo é "C"
Ou seja, UM pedaço de texto gera block_size exemplos de treino de uma vez.

Uso:
    python train.py                 # usa os hiperparâmetros padrão abaixo
    python train.py --max_iters 500 # sobrescreve qualquer hiperparâmetro pela linha de comando

Requer que `python data/prepare.py` já tenha sido executado.
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch

from model import GPT, GPTConfig

# ---------------------------------------------------------------------------
# Hiperparâmetros (padrões pensados para uma CPU/GPU modesta)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Treina um GPT char-level.")
# Dados / saída
parser.add_argument("--data_dir", default="data", help="pasta com train.bin, val.bin e meta.pkl")
parser.add_argument("--out", default="modelo.pt", help="onde salvar o checkpoint")
# Tamanho do modelo
parser.add_argument("--n_layer", type=int, default=4)
parser.add_argument("--n_head", type=int, default=4)
parser.add_argument("--n_embd", type=int, default=128)
parser.add_argument("--block_size", type=int, default=128, help="tamanho do contexto")
parser.add_argument("--dropout", type=float, default=0.1)
# Otimização
parser.add_argument("--batch_size", type=int, default=32, help="sequências por passo")
parser.add_argument("--max_iters", type=int, default=3000, help="quantos passos de treino")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--eval_interval", type=int, default=250, help="a cada quantos passos avaliar")
parser.add_argument("--eval_iters", type=int, default=50, help="batches usados para estimar a loss")
parser.add_argument("--seed", type=int, default=1337)
args = parser.parse_args()

# Semente fixa -> resultados reprodutíveis (mesmos batches, mesma inicialização).
torch.manual_seed(args.seed)

# Usa GPU se houver; caso contrário, CPU. Um modelo deste tamanho treina em
# poucos minutos na CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device}")

# ---------------------------------------------------------------------------
# 1. Carregar os dados preparados
# ---------------------------------------------------------------------------
with open(os.path.join(args.data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]

# np.memmap lê o arquivo sob demanda, sem carregar tudo na RAM (útil para
# datasets grandes; aqui é só um bom hábito).
train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")
print(f"Tokens de treino: {len(train_data):,} | validação: {len(val_data):,} | vocab: {vocab_size}")


def get_batch(split: str):
    """
    Sorteia `batch_size` posições aleatórias no texto e recorta, a partir de
    cada uma, `block_size` tokens para x e os mesmos tokens deslocados em 1 para y.

    Retorna x, y com shape (batch_size, block_size).
    """
    data = train_data if split == "train" else val_data
    # Posições iniciais aleatórias (garantindo que caiba block_size + 1 tokens).
    ix = torch.randint(len(data) - args.block_size - 1, (args.batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + args.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + args.block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# 2. Criar o modelo e o otimizador
# ---------------------------------------------------------------------------
config = GPTConfig(
    block_size=args.block_size,
    vocab_size=vocab_size,
    n_layer=args.n_layer,
    n_head=args.n_head,
    n_embd=args.n_embd,
    dropout=args.dropout,
)
model = GPT(config).to(device)

# AdamW = Adam (passo adaptativo por parâmetro, usando médias móveis do
# gradiente e do seu quadrado) + "weight decay" desacoplado (puxa os pesos
# levemente para zero, uma forma de regularização). É o padrão para Transformers.
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)


@torch.no_grad()  # não precisamos de gradientes só para medir a loss
def estimate_loss() -> dict[str, float]:
    """
    Estima a loss média em vários batches de treino e validação.
    Um único batch é muito ruidoso; a média de eval_iters batches é mais estável.
    """
    out = {}
    model.eval()  # desliga o dropout durante a avaliação
    for split in ("train", "val"):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()  # volta ao modo de treino
    return out


# ---------------------------------------------------------------------------
# 3. O loop de treino
# ---------------------------------------------------------------------------
# Referência: um modelo que chuta uniformemente tem loss = ln(vocab_size).
print(f"Loss inicial esperada (chute aleatório): {np.log(vocab_size):.2f}")

t0 = time.time()
for it in range(args.max_iters + 1):

    # Avaliação periódica (e no último passo).
    if it % args.eval_interval == 0 or it == args.max_iters:
        losses = estimate_loss()
        print(
            f"passo {it:5d} | loss treino {losses['train']:.4f} | loss val {losses['val']:.4f}"
            f" | {time.time() - t0:.0f}s"
        )

    if it == args.max_iters:
        break

    # (a) Um batch de dados.
    x, y = get_batch("train")

    # (b) Forward: previsões + loss.
    _, loss = model(x, y)

    # (c) Backward: zera gradientes antigos e calcula os novos.
    #     set_to_none=True é só um pouco mais eficiente que zerar com zeros.
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # (d) Limita o tamanho do gradiente para evitar "explosões" ocasionais.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # (e) Atualiza os pesos.
    optimizer.step()

# ---------------------------------------------------------------------------
# 4. Salvar o modelo
# ---------------------------------------------------------------------------
# Salvamos os pesos (state_dict) E a configuração, para que generate.py consiga
# reconstruir exatamente a mesma arquitetura antes de carregar os pesos.
checkpoint = {"model": model.state_dict(), "config": config, "meta": meta}
torch.save(checkpoint, args.out)
print(f"Modelo salvo em {args.out}")
