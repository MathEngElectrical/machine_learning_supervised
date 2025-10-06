from scipy.io import loadmat
import sympy as sp
import numpy as np
from defines import *
def main():

    data = loadmat(file_IN_OUT_PA)
    array_in, array_out = np.array([data_in for data_in in data['in']]), np.array([data_out for data_out in data['out']])
    X = func_calculate_array_in_volterra(array_in, P, M)
    COEFS = func_find_coef_volterra(X, array_out)


def func_calculate_array_in_volterra(list_in, P, M):
    """
    Função que ajusta os valores de entrada para o cálculo da modelagem por Volterra.
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
                list_in_memo.append(in_volterra ** p)
        list_in_volterra.append(list_in_memo)
    return list_in_volterra


def func_find_coef_volterra(array_in_volterra, list_out_volterra):
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


if __name__ == "__main__":
    main()