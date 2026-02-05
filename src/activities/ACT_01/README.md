# Atividade 1 — Regressão Linear (Mínimos Quadrados / MSE)

> **Nota sobre fórmulas:** o preview padrão do VS Code não renderiza LaTeX. Para ver as fórmulas, use uma extensão como **Markdown Preview Enhanced** ou visualize no GitHub/um renderizador que suporte MathJax/KaTeX. Este README usa delimitadores `$...$` e `$$...$$`.


Este notebook implementa **regressão linear simples** (uma variável) para ajustar a melhor reta
que relaciona pares de dados $(x_i, y_i)$, minimizando o **erro quadrático médio (MSE)**.
Além do ajuste, o notebook plota o conjunto **real** vs **estimado**.

---

## Problema matemático

Dado um conjunto de $n$ amostras $\{(x_i, y_i)\}_{i=1}^{n}$, busca-se uma reta

$$
\hat{y}_i = w\,x_i + b
$$

onde:

- $w$ é o **coeficiente angular** (inclinação),
- $b$ é o **coeficiente linear** (intercepto).

O critério de ajuste usado é o **MSE (Mean Squared Error)**:

$$
\mathrm{MSE}(w,b) = \frac{1}{n}\sum_{i=1}^{n}\left(y_i - \hat{y}_i\right)^2
= \frac{1}{n}\sum_{i=1}^{n}\left(y_i - (w x_i + b)\right)^2
$$

Como $1/n$ é constante, minimizar MSE equivale a minimizar a **soma dos quadrados do erro (SSE)**:

$$
\mathrm{SSE}(w,b)=\sum_{i=1}^{n}\left(y_i - (w x_i + b)\right)^2
$$

---

## Solução analítica por Cálculo (derivação em $w$ e $b$)

A função custo $\mathrm{SSE}(w,b)$ é convexa em $(w,b)$. Assim, o mínimo global ocorre quando:

$$
\frac{\partial \mathrm{SSE}}{\partial w} = 0
\quad\text{e}\quad
\frac{\partial \mathrm{SSE}}{\partial b} = 0
$$

Ao expandir e derivar, obtém-se o sistema (equações normais da regressão simples):

$$
\sum_{i=1}^{n} x_i\,(y_i - w x_i - b) = 0
\quad\text{e}\quad
\sum_{i=1}^{n} (y_i - w x_i - b) = 0
$$

Isolando $w$ e $b$, chega-se às fórmulas fechadas (para 1D):

$$
w = \frac{n\sum x_i y_i - (\sum x_i)(\sum y_i)}{n\sum x_i^2 - (\sum x_i)^2}
$$

$$
b = \bar{y} - w\,\bar{x}
$$

---

## Solução por Álgebra Linear (Equação Normal)

Reescrevendo o problema na forma matricial, para $d$ variáveis (generalização):

- $X \in \mathbb{R}^{n\times d}$ contém as features,
- adiciona-se uma coluna de 1’s para o bias:
  $$
  \tilde{X} = [X\ \ \mathbf{1}]
  $$
- $Y\in\mathbb{R}^{n\times 1}$.

O problema é:

$$
\min_{\theta}\ \|Y - \tilde{X}\theta\|^2
\quad\text{com}\quad
\theta=
\begin{bmatrix}
w \\ b
\end{bmatrix}
$$

A solução analítica (quando $\tilde{X}^T\tilde{X}$ é invertível) é:

$$
\theta = (\tilde{X}^T\tilde{X})^{-1}\tilde{X}^T Y
$$

Se não for invertível (matriz singular ou mal condicionada), usa-se a **pseudoinversa de Moore–Penrose**:

$$
\theta = \tilde{X}^{+}Y
$$


---

## Exemplo numérico

Dados:

- `list_x_real = [0.0, 0.1, 0.2, 0.3, 0.4]`
- `list_y_real = [0.2, 0.3, 0.45, 0.7, 0.8]`

O ajuste por mínimos quadrados resulta em:

- $w \approx 1.6$
- $b \approx 0.17$

e o erro:

$$
\mathrm{MSE} \approx 0.0012
$$

---

## O que foi implementado

Funções principais (nomes do notebook):

- `f_expr_somat_symb(list_x_real, list_y_real)`  
  Monta $\mathrm{SSE}(w,b)$ como expressão simbólica.

- `f_find_coef_analytically(exp)`  
  Calcula $w$ e $b$ resolvendo $\partial\mathrm{SSE}/\partial w = 0$ e $\partial\mathrm{SSE}/\partial b = 0$.

- `f_find_coef_numerically(list_x_real, list_y_real)`  
  Calcula $w$ e $b$ via equação normal (álgebra linear).

- `f_calculate_y_estimate(x_real_list, w, b)`  
  Gera $\hat{y}$ para cada $x$ usando $\hat{y}=wx+b$.

- `f_plot_xy(xs, ys, labels=...)`  
  Plota séries reais e estimadas com Matplotlib.

---



## Observações matemáticas importantes

- **Convexidade:** SSE/MSE é uma função quadrática em $(w,b)$, logo tem mínimo global único (exceto em casos degenerados).
- **Condição de invertibilidade:** para a equação normal, precisa-se que $\tilde{X}^T\tilde{X}$ seja invertível; caso contrário, a pseudoinversa é o caminho robusto.
- **Interpretação geométrica:** $\tilde{X}\theta$ é a projeção de $Y$ no subespaço gerado pelas colunas de $\tilde{X}$.
