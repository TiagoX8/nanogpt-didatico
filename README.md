# nanogpt-didatico

Um modelo de linguagem tipo **GPT**, implementado **do zero em PyTorch**, em menos de
400 linhas comentadas em português. O objetivo não é bater recordes: é **entender
como um LLM funciona** lendo (e mexendo em) cada peça.

Inspirado no [nanoGPT](https://github.com/karpathy/nanoGPT) e na aula
["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) do Andrej Karpathy.

O modelo é treinado em *Dom Casmurro* (Machado de Assis, domínio público) e, ao final,
gera texto novo "no estilo" do livro, um caractere por vez.

## Estrutura

```
├── README.md
├── requirements.txt
├── data/
│   ├── dom_casmurro.txt   # texto bruto (Project Gutenberg, domínio público)
│   └── prepare.py         # tokenização char-level -> train.bin, val.bin, meta.pkl
├── model.py               # a arquitetura GPT (embeddings, atenção, MLP, blocos, cabeça)
├── train.py               # loop de treino: batches, loss, backprop, AdamW, checkpoint
└── generate.py            # carrega o checkpoint e gera texto por amostragem
```

Sugestão de leitura: `data/prepare.py` → `model.py` → `train.py` → `generate.py`.

## Instalação

Requer Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Sem GPU? Sem problema. Para instalar o PyTorch só com CPU (download menor):
> `pip install torch --index-url https://download.pytorch.org/whl/cpu numpy`

## Como usar

### 1. Preparar os dados

```bash
python data/prepare.py
```

Lê `data/dom_casmurro.txt`, descobre o vocabulário (~94 caracteres distintos),
converte o texto em inteiros e salva `train.bin`, `val.bin` e `meta.pkl` em `data/`.

### 2. Treinar

```bash
python train.py
```

Com os padrões (4 camadas, 4 cabeças, 128 dimensões, ~0,8M parâmetros, 3000 passos)
leva alguns minutos em CPU. A loss começa perto de `ln(94) ≈ 4,5` (chute aleatório)
e deve cair para cerca de `1,5`. O modelo é salvo em `modelo.pt`.

Todos os hiperparâmetros podem ser alterados pela linha de comando:

```bash
python train.py --max_iters 500                      # treino rápido só para testar
python train.py --n_layer 6 --n_embd 256 --max_iters 5000   # modelo maior (melhor com GPU)
```

### 3. Gerar texto

```bash
python generate.py
python generate.py --prompt "Capitú " --max_new_tokens 300 --temperature 0.7
```

Experimente variar `--temperature` (0,5 = conservador, 1,2 = caótico) e `--top_k`.

## Conceitos em 5 minutos

### Tokenização

Redes neurais só operam com números. **Tokenizar** é converter texto em uma sequência
de inteiros. Aqui usamos o esquema mais simples: **cada caractere é um token**
(`"ola"` → `[58, 55, 44]`). LLMs reais usam *subpalavras* (BPE), em que um token pode
ser um pedaço de palavra, o que encurta as sequências. A ideia é a mesma.

### Embeddings

Um inteiro não carrega significado. O **embedding** é uma tabela `vocab_size × n_embd`
que troca cada token por um vetor de números reais, **aprendido durante o treino**.
Tokens usados em contextos parecidos acabam com vetores parecidos. Como a atenção não
sabe a ordem dos tokens, somamos também um **embedding de posição** (um vetor para
"posição 0", outro para "posição 1", ...).

### Self-attention

É o mecanismo que permite aos tokens **trocar informação entre si**. Cada token produz
três vetores: *query* ("o que procuro?"), *key* ("o que ofereço?") e *value* ("o que
carrego"). A afinidade entre dois tokens é `query · key`; um softmax transforma essas
afinidades em pesos, e a saída de cada token é a média ponderada dos *values* dos outros.

- **Causal**: uma máscara triangular impede um token de olhar para o futuro
  (senão prever o próximo token seria trapaça).
- **Multi-head**: várias atenções pequenas em paralelo, cada uma livre para aprender
  um "tipo" de relação diferente.

Depois da atenção, um **MLP** (feed-forward) processa cada token individualmente.
Atenção + MLP, com *layer norm* e conexões residuais, formam um **bloco Transformer**;
o GPT é uma pilha desses blocos seguida de uma camada linear que produz um número
(*logit*) para cada token possível como próximo.

### Loop de treino

1. Sorteia pedaços aleatórios do texto: `x` (entrada) e `y` (o mesmo pedaço deslocado
   em 1 — a "resposta certa" de cada posição).
2. **Forward**: o modelo prevê o próximo token para cada posição de `x`.
3. **Loss** (*cross-entropy*): `-log(probabilidade dada ao token correto)`. Quanto mais
   confiante e certo, menor a loss.
4. **Backward** (*backpropagation*): `loss.backward()` calcula quanto cada peso
   contribuiu para o erro.
5. **Otimizador** (*AdamW*): dá um passo pequeno em cada peso na direção que reduz a loss.
6. Repete milhares de vezes. Periodicamente mede a loss em dados de **validação**, que o
   modelo nunca viu, para verificar que está generalizando e não decorando.

### Geração

Dado um prompt, o modelo calcula as probabilidades do próximo caractere, **sorteia**
um (amostragem), anexa ao texto e repete. A **temperatura** divide os logits antes do
softmax: valores baixos deixam a distribuição mais "afiada" (escolhas seguras), valores
altos a deixam mais uniforme (mais surpresa).

## Ideias para continuar aprendendo

- Troque `data/dom_casmurro.txt` por outro texto (letras de música, código, seus e-mails).
- Aumente `--n_layer`, `--n_embd` e `--max_iters` e veja a loss de validação cair.
- Remova a máscara causal em `model.py` e observe a loss "trapacear" e ir a quase zero.
- Implemente um tokenizador BPE no lugar do char-level.
- Leia o código do [nanoGPT](https://github.com/karpathy/nanoGPT) original e compare.
