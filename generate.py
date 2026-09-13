"""
generate.py — Gerar texto com o modelo treinado.

Aqui acontece a "mágica" que se vê no ChatGPT: o modelo recebe um começo de
texto (prompt) e vai escrevendo um caractere por vez.

Como funciona cada passo:
    1. Codifica o texto atual em inteiros.
    2. Passa pelo modelo -> logits para o próximo token.
    3. Aplica temperatura / top-k e transforma em probabilidades (softmax).
    4. SORTEIA um token de acordo com essas probabilidades (amostragem).
    5. Anexa o token ao texto e volta ao passo 1.

Detalhe importante: o modelo NÃO "sabe" o que vai escrever. Ele só prevê o
próximo caractere, repetidamente. Toda a coerência que aparece emerge de ter
aprendido bem essa única tarefa.

Uso:
    python generate.py
    python generate.py --prompt "Capitú " --max_new_tokens 500 --temperature 0.8
"""

import argparse

import torch

from model import GPT

parser = argparse.ArgumentParser(description="Gera texto com um GPT char-level treinado.")
parser.add_argument("--checkpoint", default="modelo.pt", help="arquivo salvo por train.py")
parser.add_argument("--prompt", default="\n", help="texto inicial (padrão: uma quebra de linha)")
parser.add_argument("--max_new_tokens", type=int, default=500, help="quantos caracteres gerar")
parser.add_argument(
    "--temperature", type=float, default=0.8,
    help="< 1 = mais previsível; > 1 = mais aleatório; 1.0 = probabilidades originais",
)
parser.add_argument("--top_k", type=int, default=None, help="considerar só os k tokens mais prováveis")
parser.add_argument("--num_samples", type=int, default=1, help="quantas amostras gerar")
parser.add_argument("--seed", type=int, default=None, help="semente para resultados reprodutíveis")
args = parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 1. Carregar o checkpoint
# ---------------------------------------------------------------------------
# weights_only=False porque o checkpoint contém também a GPTConfig (dataclass)
# e o vocabulário, e não apenas tensores.
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
config = checkpoint["config"]
meta = checkpoint["meta"]

# Reconstrói a arquitetura com a MESMA configuração e carrega os pesos aprendidos.
model = GPT(config)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()  # modo de avaliação: desliga o dropout

# ---------------------------------------------------------------------------
# 2. Funções de (de)codificação, a partir do vocabulário salvo em prepare.py
# ---------------------------------------------------------------------------
stoi, itos = meta["stoi"], meta["itos"]


def encode(s: str) -> list[int]:
    # Caracteres que não existem no vocabulário são ignorados: o modelo nunca
    # os viu e não teria um embedding para eles.
    return [stoi[c] for c in s if c in stoi]


def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)


# ---------------------------------------------------------------------------
# 3. Gerar
# ---------------------------------------------------------------------------
prompt_ids = encode(args.prompt)
if not prompt_ids:
    raise SystemExit("O prompt não contém nenhum caractere conhecido pelo modelo.")

# Shape (1, T): batch de 1 sequência.
x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]

for i in range(args.num_samples):
    y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(decode(y[0].tolist()))
    print("-" * 60)
