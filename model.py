"""
model.py — Um GPT pequeno, implementado do zero em PyTorch.

GPT significa "Generative Pre-trained Transformer". Na prática, é um modelo
que recebe uma sequência de tokens e devolve, para CADA posição, uma
distribuição de probabilidade sobre qual será o PRÓXIMO token.

Visão geral da arquitetura (de baixo para cima):

    tokens (inteiros)
        │
        ▼
    Embedding de tokens  +  Embedding de posições      -> vetores de tamanho n_embd
        │
        ▼
    Bloco Transformer  ×  n_layer
        ├─ LayerNorm -> Multi-Head Self-Attention -> soma residual
        └─ LayerNorm -> Feed-Forward (MLP)        -> soma residual
        │
        ▼
    LayerNorm final
        │
        ▼
    Cabeça linear (n_embd -> vocab_size)               -> "logits"
        │
        ▼
    softmax -> probabilidade do próximo token

Cada componente está implementado como uma classe nn.Module abaixo, com
comentários explicando o que faz e por que existe.

Convenção de dimensões usada nos comentários:
    B = batch size       (quantas sequências processamos ao mesmo tempo)
    T = tamanho da sequência (quantos tokens de contexto; T <= block_size)
    C = n_embd           (tamanho do vetor que representa cada token)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    """Hiperparâmetros que definem o TAMANHO e a FORMA do modelo."""

    block_size: int = 256   # contexto máximo: quantos tokens o modelo "enxerga" de cada vez
    vocab_size: int = 100   # quantidade de tokens distintos (definido pelos dados)
    n_layer: int = 6        # quantidade de blocos Transformer empilhados
    n_head: int = 6         # quantidade de "cabeças" de atenção em cada bloco
    n_embd: int = 384       # dimensão dos vetores internos (deve ser divisível por n_head)
    dropout: float = 0.2    # regularização: zera aleatoriamente ativações durante o treino
    bias: bool = False      # usar bias nas camadas Linear / LayerNorm? (GPT-2 usa; sem é um pouco melhor)


# ---------------------------------------------------------------------------
# Self-Attention com múltiplas cabeças
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """
    O coração do Transformer.

    A ideia: cada token produz três vetores a partir do seu embedding:
        Q (query)  — "o que eu estou procurando?"
        K (key)    — "o que eu tenho a oferecer?"
        V (value)  — "a informação que eu carrego"

    A afinidade entre o token i e o token j é o produto escalar  Q_i · K_j.
    Aplicamos softmax nessas afinidades para obter PESOS que somam 1, e a
    saída do token i é a média ponderada dos V_j. Assim cada token "recolhe"
    informação dos tokens relevantes para ele.

    "Causal" significa que o token i só pode olhar para tokens j <= i (passado).
    Sem isso, o modelo veria o futuro e a tarefa de prever o próximo token
    seria trivial. Implementamos isso com uma MÁSCARA triangular.

    "Multi-head": em vez de uma única atenção de tamanho C, fazemos n_head
    atenções independentes de tamanho C/n_head e concatenamos. Cada cabeça
    pode aprender a olhar para coisas diferentes (ex.: uma cuida de
    concordância, outra de pontuação, outra do token imediatamente anterior).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd deve ser divisível por n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Uma única camada Linear produz Q, K e V de todas as cabeças de uma vez
        # (3 * n_embd saídas). É apenas uma otimização: equivale a três Linear separadas.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Projeção de saída: mistura as informações vindas das várias cabeças.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Máscara causal: matriz triangular inferior de 1s.
        # register_buffer = faz parte do módulo (é salvo, vai para a GPU junto),
        # mas NÃO é um parâmetro treinável.
        #   [[1, 0, 0],
        #    [1, 1, 0],
        #    [1, 1, 1]]  -> a linha i (token i) só vê colunas j <= i.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        hs = C // self.n_head  # head size: dimensão de cada cabeça

        # 1. Projetar x em Q, K, V.        (B, T, C) -> (B, T, 3C) -> 3 × (B, T, C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 2. Separar as cabeças: (B, T, C) -> (B, T, n_head, hs) -> (B, n_head, T, hs)
        #    Agora cada cabeça é tratada como uma "dimensão de batch" extra.
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # 3. Afinidades: Q · K^T  -> (B, n_head, T, T). att[b, h, i, j] = quanto o token i
        #    "presta atenção" no token j. Dividimos por sqrt(hs) para que os valores não
        #    fiquem enormes (o que deixaria o softmax "afiado" demais e o gradiente ruim).
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)

        # 4. Máscara causal: onde mask == 0 (futuro), colocamos -infinito.
        #    Após o softmax, exp(-inf) = 0, ou seja, peso zero para o futuro.
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # 5. Softmax transforma as afinidades em pesos que somam 1 em cada linha.
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 6. Média ponderada dos valores: (B, n_head, T, T) @ (B, n_head, T, hs) -> (B, n_head, T, hs)
        y = att @ v

        # 7. Juntar as cabeças de volta: (B, n_head, T, hs) -> (B, T, n_head, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Projeção final + dropout.
        return self.resid_dropout(self.c_proj(y))


# ---------------------------------------------------------------------------
# Feed-Forward (MLP)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """
    Rede feed-forward aplicada a CADA TOKEN INDEPENDENTEMENTE.

    Se a atenção é onde os tokens "conversam" entre si, o MLP é onde cada
    token "pensa sozinho" sobre o que recolheu. Expande para 4× a dimensão,
    aplica uma não-linearidade (GELU) e volta para a dimensão original.
    É aqui que boa parte do "conhecimento" do modelo fica armazenada.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()  # parecida com ReLU, mas suave; padrão nos GPTs
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


# ---------------------------------------------------------------------------
# Bloco Transformer
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """
    Um bloco = comunicação (atenção) + computação (MLP), cada um com:

      * LayerNorm ANTES da sub-camada ("pre-norm", como no GPT-2). O LayerNorm
        normaliza cada vetor de token para média 0 e variância 1, o que
        estabiliza o treino de redes profundas.

      * Conexão RESIDUAL:  x = x + sublayer(x). Em vez de substituir x, a
        sub-camada apenas ADICIONA uma correção. Isso cria um "caminho direto"
        para o gradiente fluir da saída até as primeiras camadas, permitindo
        empilhar muitos blocos sem que o treino trave.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))  # tokens trocam informação
        x = x + self.mlp(self.ln_2(x))   # cada token processa o que recebeu
        return x


# ---------------------------------------------------------------------------
# O modelo completo
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Embedding de TOKENS: uma tabela vocab_size × n_embd. A linha i é o
        # vetor que representa o token i. Esses vetores são APRENDIDOS: tokens
        # que aparecem em contextos parecidos acabam com vetores parecidos.
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Embedding de POSIÇÕES: a atenção, por si só, não sabe a ORDEM dos
        # tokens (é uma soma ponderada, que é comutativa). Então somamos a cada
        # token um vetor que depende da sua posição (0, 1, 2, ..., T-1).
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)

        # Cabeça de saída: transforma o vetor de cada token em vocab_size
        # números ("logits"), um para cada token possível como próximo.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # "Weight tying": a cabeça de saída reutiliza a MESMA matriz do embedding
        # de entrada. Faz sentido (ambas mapeiam entre tokens e vetores),
        # economiza parâmetros e costuma melhorar os resultados.
        self.lm_head.weight = self.wte.weight

        # Inicialização dos pesos (mesma receita do GPT-2).
        self.apply(self._init_weights)
        # As projeções residuais recebem um desvio-padrão menor, escalado pelo
        # número de camadas, para que a soma de muitos resíduos não exploda.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"Modelo criado com {self.num_params() / 1e6:.2f}M parâmetros")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Total de parâmetros treináveis (o embedding de posição incluído)."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx:     (B, T) inteiros — os tokens de entrada.
        targets: (B, T) inteiros — os tokens "corretos" seguintes (opcional).

        Retorna (logits, loss). loss é None se targets não for fornecido.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequência de {T} tokens excede block_size={self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)

        tok_emb = self.wte(idx)   # (B, T, C)  -> "o que é cada token"
        pos_emb = self.wpe(pos)   # (T, C)     -> "onde está cada token"
        x = self.drop(tok_emb + pos_emb)  # broadcasting soma pos_emb a cada exemplo do batch

        for block in self.blocks:
            x = block(x)          # (B, T, C)
        x = self.ln_f(x)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy compara, para cada posição, a distribuição prevista
            # com o token correto. Ela é  -log(p[token_correto]): se o modelo
            # deu probabilidade alta ao token certo, a loss é baixa.
            # F.cross_entropy espera (N, classes) e (N,), então "achatamos" B e T.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Gera texto de forma AUTORREGRESSIVA: prevê um token, anexa-o à
        sequência, prevê o próximo, e assim por diante.

        idx:            (B, T) tokens iniciais (o "prompt").
        max_new_tokens: quantos tokens novos gerar.
        temperature:    < 1 deixa o modelo mais "conservador" (escolhe os tokens
                        mais prováveis); > 1 deixa mais "criativo"/aleatório.
        top_k:          se definido, só considera os k tokens mais prováveis.
        """
        for _ in range(max_new_tokens):
            # O modelo só enxerga block_size tokens: cortamos o começo se preciso.
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)

            # Só nos interessa a previsão da ÚLTIMA posição (o próximo token).
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            if top_k is not None:
                # Zera (coloca -inf) tudo que não está entre os k maiores logits.
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Logits -> probabilidades.
            probs = F.softmax(logits, dim=-1)

            # AMOSTRAGEM: sorteamos o próximo token de acordo com as probabilidades.
            # (Se sempre pegássemos o argmax, o texto ficaria repetitivo.)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Anexa o token gerado e repete.
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
