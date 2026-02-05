# ATIVIDADE 2 - Modelagem de amplificador via Volterra Série (Memory Polynomial)


Implementação de um modelo não linear com **memória** a partir de dados reais de laboratório (entrada/saída de um amplificador). Baseado na série de Volterra, o modelo é uma forma simplificada conhecida como **Memory Polynomial (MP)**.

---

Exemplo do `main()` (trecho simplificado):

---

## O modelo: Volterra “simplificado” (Memory Polynomial)

O conjunto de termos construído é um **Memory Polynomial (MP)**, que pode ser visto como uma forma “diagonal/truncada” da série de Volterra.

Parâmetros

- **P** → grau do polinômio (ordem não linear)
- **M** → memória (quantos instantes passados influenciam o presente)

### Vetor de regressores (features)

Para cada amostra no tempo `n`, montamos o vetor de regressores φ(n) com potências e atrasos de x:

$$
\phi(n)=\big[
x(n)^1,\;x(n-1)^1,\;\ldots,\;x(n-M)^1,\;
x(n)^2,\;\ldots,\;x(n-M)^2,\;
\ldots,\;
x(n)^P,\;\ldots,\;x(n-M)^P
\big]
$$

O modelo estimado (incluindo bias/intercepto) pode ser escrito como:

$$
\hat{y}(n)= b + \sum_{p=1}^{P}\sum_{m=0}^{M} a_{p,m}\,x(n-m)^p
$$

Onde:

- $a_{p,m}$ são os coeficientes do termo de ordem $p$ e atraso $m$;
- $b$ é o intercepto (bias);

---

## Construção da matriz X — `f_calculate_array_in_volterra`

###  O que a função produz

A função transforma a série de entrada em uma **matriz de regressão**:

- $X \in \mathbb{R}^{N \times K}$;
- $N$ → número de amostras;
- $K = P\,(M+1)$ → número de features por amostra;

Cada linha de `X` é o vetor φ(n).

## Calculando coeficiente — `f_find_coef_volterra`

Essa função é o **núcleo do método**: ela estima os coeficientes do modelo resolvendo um problema de **mínimos quadrados**.

### Do ponto de vista matemático

Uma vez montados:

- $X \in \mathbb{R}^{N\times K}$ (features / regressores)
- $y \in \mathbb{R}^{N\times 1}$ (saída medida)

Queremos encontrar $theta$ que minimize o erro quadrático:

$$
\min_{\theta}\; \lVert y - X\theta \rVert_2^2
$$

Como o código também estima o **intercepto** \(b\), ele trabalha com uma matriz estendida:

$$
\tilde{X} = [X\;\;\mathbf{1}]
$$

onde $\mathbf{1}$ é uma coluna de 1s (shape $N\times 1$).  
Então o vetor de parâmetros passa a incluir o bias:

$$
\theta =
\begin{bmatrix}
a_{1,0} & \cdots & a_{P,M} & b
\end{bmatrix}^{\top}
$$

A solução de mínimos quadrados é obtida via **pseudo-inversa** (Moore–Penrose):

$$
\theta = \tilde{X}^{+}y
$$

### Matemática traduzida em programação

A implementação do notebook (equivalente ao `f_find_coef_volterra` do repositório) segue, passo a passo, as equações acima:

#### (1) Garantir que entrada e saída estão em matrizes NumPy 2D

Os dados chegam como listas/arrays que podem ter “sublistas” ou shapes `(N,1)`.  
O código padroniza isso usando `np.array(...)` + `flatten()` por linha.

Ideia:

```python
in_volterra  = np.array([np.array(x).flatten() for x in array_in_volterra])  # -> X  (N x K)
out_volterra = np.array([np.array(y).flatten() for y in list_out_volterra])  # -> y  (N x 1)
```

**Tradução matemática:** montar `X` e `y` com shapes compatíveis para álgebra linear.

#### (2) Adicionar o termo de bias com uma coluna de 1s

```python
X_adjust = np.hstack([in_volterra, np.ones((in_volterra.shape[0], 1))])
```

**Tradução matemática:** construir \(\tilde{X}=[X\;\;\mathbf{1}]\).

#### (3) Resolver mínimos quadrados usando pseudo-inversa

```python
COEFS = np.linalg.pinv(X_adjust) @ out_volterra
```

**Tradução matemática:** implementar $\theta = \tilde{X}^{+}y$ diretamente.

- `np.linalg.pinv(·)` retorna $\tilde{X}^{+}$
- `@` faz multiplicação matricial

#### (4) Como interpretar `COEFS`

Como a coluna de 1s foi adicionada **por último**, o vetor final contém:

- `COEFS[:-1]` → coeficientes dos termos $a_{p,m}$ (na ordem definida pelos laços de `p` e `m`)
- `COEFS[-1]`  → bias/intercepto \(b\)

> Dica prática: se você quiser “mapear” um índice de `COEFS` para um par \((p,m)\), basta lembrar que o vetor está organizado em blocos de tamanho `(M+1)` por ordem `p`.

### Pseudo-inversa

Em geral, \(X\) é **retangular** (muito mais linhas do que colunas) e pode ser **mal-condicionada** (features altamente correlacionadas, principalmente com potências e atrasos).  
A pseudo-inversa (`pinv`) é uma forma robusta de obter a solução de mínimos quadrados.

Internamente, `np.linalg.pinv` calcula a pseudo-inversa via **SVD (decomposição em valores singulares)**:

$$
\tilde{X} = U\Sigma V^{\top}
\quad\Rightarrow\quad
\tilde{X}^{+} = V\Sigma^{+}U^{\top}
$$

onde $\Sigma^{+}$ inverte apenas valores singulares “significativos” (descartando os muito pequenos por tolerância), reduzindo instabilidade numérica.


