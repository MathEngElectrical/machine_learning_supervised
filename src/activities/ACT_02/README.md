# ATIVIDADE_2 — Modelagem de um amplificador via Volterra (Memory Polynomial) em Python

Este notebook implementa uma **identificação de modelo não linear com memória** a partir de dados reais de laboratório (entrada/saída de um amplificador).  
Na prática, ele monta uma matriz de regressão com termos polinomiais e atrasos (memória) e estima os coeficientes por **mínimos quadrados** usando a **pseudo-inversa de Moore–Penrose**.

> Observação: apesar do texto citar “Volterra”, o que é implementado no código é um **Memory Polynomial (MP)** (um subconjunto/forma “diagonal” do modelo de Volterra), muito usado em modelagem comportamental de amplificadores com memória. 

---

## Dados de entrada

O notebook carrega um arquivo MATLAB:

- `IN_OUT_PA.mat` contendo duas chaves:
  - `in`  → amostras de **entrada**
  - `out` → amostras de **saída** (medidas)

## Visão geral do modelo (MP / Volterra simplificado)

O notebook define:

- **P** = grau do polinômio (ordem não-linear)
- **M** = memória (quantos instantes passados influenciam o presente)

E constrói o vetor de regressores para cada amostra `n` como:

\[
\phi(n)=
\big[
x(n)^1, x(n-1)^1, \dots, x(n-M)^1,\;
x(n)^2, x(n-1)^2, \dots, x(n-M)^2,\;
\dots,\;
x(n)^P, x(n-1)^P, \dots, x(n-M)^P
\big]
\]

O modelo estimado (com termo de bias/intercepto) fica:

\[
\hat{y}(n)= b + \sum_{p=1}^{P}\sum_{m=0}^{M} a_{p,m}\,x(n-m)^p
\]

Esse formato é conhecido como **memory polynomial** (derivado/truncado de Volterra). 

---

## Funções do notebook

### 1) `func_calculate_array_in_volterra(list_in, P, M)`

**Objetivo:** transformar a série temporal de entrada em uma lista/matriz de regressão no formato MP.

Pontos importantes:

- Para `in_index - m < 0`, o código usa `0` (zero-padding no início).
- Cada linha gerada tem `P*(M+1)` termos.
- Não há termos cruzados (ex.: `x(n)x(n-1)`), então não é o Volterra completo.

Código (do notebook):

```python
def func_calculate_array_in_volterra(list_in, P, M):
    list_in_volterra = []
    for in_index in range(len(list_in)):
        list_in_memo = []
        for p in range(1, P + 1):
            for m in range(0, M + 1):
                if in_index - m < 0:
                    in_volterra = 0
                else:
                    in_volterra = list_in[in_index - m][0]
                list_in_memo.append(in_volterra**p)
        list_in_volterra.append(list_in_memo)
    return list_in_volterra
```

---

## Foco especial: `func_find_coef_volterra(array_in_volterra, list_out_volterra)`

### O que ela faz

Essa função resolve um problema clássico de **identificação por mínimos quadrados**:

1. Converte as listas para arrays 2D:
   - `in_volterra` vira uma matriz **X** de dimensão \(N \times K\), onde:
     - \(N\) = número de amostras
     - \(K\) = número de regressores = `P*(M+1)`  
2. Converte a saída para vetor/matriz **y** (dimensão \(N\times 1\) ou \(N\times L\) se houver múltiplas saídas).
3. Adiciona uma coluna de **1s** na entrada para estimar o **bias** (intercepto) `b`.
4. Calcula os coeficientes com pseudo-inversa:

\[
\theta = X^{+}y
\]

onde \(X^{+}\) é a **pseudo-inversa de Moore–Penrose**.

Código (do notebook):

```python
def func_find_coef_volterra(array_in_volterra, list_out_volterra):
    in_volterra = np.array([np.array(x).flatten() for x in array_in_volterra])
    out_volterra = np.array([np.array(y).flatten() for y in list_out_volterra])
    in_volterra_adjust = np.hstack([in_volterra, np.ones((in_volterra.shape[0], 1))])
    COEFS = np.linalg.pinv(in_volterra_adjust) @ out_volterra
    return COEFS
```

### Interpretação matemática (mínimos quadrados)

A função está minimizando o erro quadrático entre o valor medido e o estimado:

\[
\min_{\theta}\; \|y - X\theta\|_2^2
\]

- Se \(X\) fosse quadrada e invertível, teríamos \(\theta=X^{-1}y\).
- Em problemas reais, \(X\) costuma ser **retangular** (mais amostras do que parâmetros) e/ou **mal-condicionada**, então usamos mínimos quadrados.

Uma forma equivalente (quando \(X\) tem posto completo) é a **equação normal**:

\[
\theta = (X^T X)^{-1}X^T y
\]

Mas isso pode ser numericamente instável. A pseudo-inversa via SVD costuma ser mais robusta.

### Como o NumPy implementa `pinv`

`np.linalg.pinv(X)` calcula a pseudo-inversa de Moore–Penrose usando **decomposição em valores singulares (SVD)**. 

Em termos de álgebra linear:

1. Faz a SVD (economy):  
\[
X = U\Sigma V^T
\]

2. Inverte apenas os valores singulares “significativos” (corte por tolerância), formando \(\Sigma^+\).

3. Retorna:  
\[
X^+ = V\Sigma^+U^T
\]

O parâmetro `rcond`/`rtol` define o **corte para valores singulares pequenos**, que são tratados como zero para evitar explosões numéricas. 

> Por que isso é importante aqui?  
> Modelos MP/Volterra podem gerar colunas muito correlacionadas (ex.: `x`, `x^2`, `x^3` e atrasos), o que piora o condicionamento. A SVD com cutoff ajuda a estabilizar a solução.

### Por que adicionar o bias (coluna de 1s)

A linha:

```python
in_volterra_adjust = np.hstack([in_volterra, np.ones((in_volterra.shape[0], 1))])
```

inclui o termo \(b\) no modelo. Isso permite que o ajuste encontre um deslocamento constante na saída, mesmo quando todos os termos polinomiais são zero (por exemplo, no padding inicial).

---

## Execução no notebook

A sequência central é:

```python
X = func_calculate_array_in_volterra(array_in, P, M)
COEFS = func_find_coef_volterra(X, array_out)
X = np.array(X)
```

E depois:

- Confere o shape de `in_volterra`
- Imprime os coeficientes estimados

---

## Observação sobre a última célula (rascunho)

A célula final do notebook tem:

```python
out_array = np.array([np.array(y).flatten() for y in lista_out])
X_bias = np.hstack([in_volterra, np.ones((in_volterra.shape[0], 0))])
COEFS = np.linalg.pinv(X_bias) @ out_array
```

Do jeito que está:

- `lista_out` **não aparece definido** no notebook (provável intenção era usar `array_out`).
- `np.ones((..., 0))` cria **zero colunas**, então **não adiciona bias** (provável que o correto fosse `1`).

Como a função `func_find_coef_volterra` já resolve isso corretamente, essa célula parece um **teste/rascunho** e pode ser removida ou corrigida.

---

## Próximos passos sugeridos

- **Validação do modelo:** calcular \(\hat{y}\) e métricas (MSE, NMSE, R²) em treino/teste.
- **Normalização/escala:** padronizar `x` para melhorar condicionamento antes de elevar potências.
- **Regularização (Ridge):** se houver instabilidade, resolver  
  \(\min \|y-X\theta\|^2 + \lambda\|\theta\|^2\).
- **Volterra mais completo (termos cruzados):** incluir produtos entre atrasos para capturar efeitos não-lineares mais gerais (à custa de muito mais parâmetros). 

---

## Referências (para aprofundar)

- Documentação do **NumPy**: `numpy.linalg.pinv` (pseudo-inversa via SVD e cutoff de valores singulares).
- Visão geral de **SVD** e pseudo-inversa de Moore–Penrose (álgebra linear aplicada a mínimos quadrados).
- Documentação/leituras sobre **Memory Polynomial** derivado de **Volterra** para modelagem de amplificadores (MP / GMP).
- Artigos e notas de modelagem comportamental de amplificadores com efeitos de memória (Volterra, MP, GMP).
