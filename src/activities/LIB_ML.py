"""
Módulo com funções para estudo de Machine Learning no contexto de Amplificadores de potência.
-------------------------------------------
Este módulo contém funções utilitárias para o pipeline das atividades propostas.
-------------------------------------------
Autor: Matheus Dias
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def f_expr_somat_symb(list_x_real, list_y_real):
    """
    Função que desenvolve uma expressão de somatório simbolicamente para equação da reta.
    Args:
        list_x_real → Lista com os valores de entrada reais do sistema. (List)
        list_y_real → Lista com os valores de saída reais do sistema. (List)
    Return:
        exp_resul → Expressão simbólica. (Sympy tipo)
    """
    w, b = sp.symbols('w b')
    exp_resul = 0
    for i_y, x_real in enumerate(list_x_real):
        y = list_y_real[i_y]
        exp_resul += (y - (x_real*w + b))**2
    exp_resul = sp.simplify(exp_resul)
    return exp_resul


def f_find_coef_analytically(exp):
    """
    Função que encontra os coeficientes especificamente para modelagem com a equação da reta.
    Args:
        exp → Expressão simbólica. (Sympy tipo)
    Returns:
        w, b → Coeficientes w e b do modelo. (Dict)
    """
    w, b = sp.symbols('w b')
    eq_diff_w, eq_diff_b = sp.Eq(sp.diff(exp, w), 0), sp.Eq(sp.diff(exp, b), 0)
    COEFS = sp.solve([eq_diff_w, eq_diff_b], (w, b))
    w, b = COEFS[w], COEFS[b]
    return w, b


def f_find_coef_numerically(list_x_real, list_y_real):
    """
    Função que encontra os coeficientes angular e linear que são ótimos usando métodos otimizados da lib numpy.
    Args:
        Dados reais de entrada e saída.
        list_x_real → Lista com as entradas. (List)
        list_y_real → Lista com as saídas. (List)
    Returns:
        String com os valores dos coeficientes.
    Obs:
        O caso Penrose não foi implementado.
    """
    define_index_line = 0
    X, Y = np.array(list_x_real).reshape(-1, 1), np.array(list_y_real).reshape(-1, 1)
    XX = np.hstack((X, np.ones((X.shape[define_index_line], 1))))
    COEFS = np.linalg.inv(XX.T @ XX) @ XX.T @ Y # IMPLEMENTA A "EQUAÇÃO NORMAL"
    w, b = COEFS[0][0], COEFS[1][1]
    return w, b


def f_calculate_y_estimate(list_x_real, w, b):
    """
    Função que calcula a saída estimada conforme os parametros calculados.
    Args:
        list_x_real → Lista com valores reais. (list)
        w → Coeficiente angular. (float)
        b → Coeficiente linear. (float)
    Returns:
        Lista com valores estimados.
    """
    return [w*x + b for x in list_x_real]


def f_plot_xy(xs, ys, labels=None, x_label="X", y_label="Y", title=None):
    """
    Plota múltiplas curvas X vs Y num único gráfico usando Matplotlib.

    Cada par (x, y) em xs e ys representa uma curva. Opcionalmente,
    rótulos podem ser fornecidos para identificação na legenda.

    Args:
        xs → Lista contendo os valores do eixo X para cada curva. (list array-like)
        ys → Lista contendo os valores do eixo Y correspondentes aos valores em xs. (list array-like)

        labels → Lista de rótulos para cada curva. Deve ter o mesmo comprimento de `xs` e `ys`. Se None, as curvas
                  serão plotadas sem rótulo.
        x_label → Rótulo do eixo X. Padrão é "X".
        y_label → Rótulo do eixo Y. Padrão é "Y".
        title → Título do gráfico. Se None, nenhum título erá exibido.
    Raises:
        ValueError: Se `labels` for fornecido e seu tamanho não corresponder
            ao tamanho de `xs` e `ys`.
    Returns:
        None
    """
    if labels is not None and not (len(xs) == len(ys) == len(labels)):
        raise ValueError("xs, ys e labels devem ter o mesmo tamanho")

    for x, y, label in zip(xs, ys, labels or [None] * len(xs)):
        plt.plot(x, y, label=label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if title:
        plt.title(title)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def f_calculate_array_in_volterra(list_in, P, M):
    """
    Função que ajusta os valores de entrada para o cálculo da modelagem por volterra.
    Args:
        list_in → Valores de entrada que devem ser ajustados. (List)
        P → Grau do polinômio modelador. (Int)
        M → Memória do sistema que configura quantas entradas passadas a atual depende. (Int)
    Returns:
        list_in_volterra → Lista com os valores de entrada ajustados. (List)
    """
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


def f_find_coef_volterra(array_in_volterra, list_out_volterra):
    """
    Função que encontra os coeficientes que minimizam a soma das distâncias entre valores reais e estimados.
    Args:
        array_in_volterra → Lista de entradas ajustadas pelo modelo de Volterra. (List)
        list_out_volterra → Lista de saídas reais observadas. (List)
    Returns:
        COEFS → Vetor de coeficientes estimados pelo modelo de Volterra. (np.ndarray)
    """
    in_volterra = np.array([np.array(x).flatten() for x in array_in_volterra])
    out_volterra = np.array([np.array(y).flatten() for y in list_out_volterra])
    in_volterra_adjust = np.hstack([in_volterra, np.ones((in_volterra.shape[0], 1))])
    COEFS = np.linalg.pinv(in_volterra_adjust) @ out_volterra
    return COEFS


def f_show_coefs(coefs):
    """
    Função que formata os coeficientes para exibição.
    Args:
        coefs → Vetor de coeficientes. (np.ndarray)
    Returns:
        str_coefs → String formatada com os coeficientes. (str)
    """
    str_coefs = ', '.join([f'c{i+1}={coef[0]:.6f}' for i, coef in enumerate(coefs)])
    return str_coefs