from scipy.io import loadmat
import sympy as Symbol
import numpy as np
from math import log10
from defines import *
from implement_volterra import func_find_coef_volterra
import matplotlib.pyplot as plt

def main():

    data = loadmat(file_in_out_SBRT2_direto)
    list_in_complex, lista_out_complex = [data_in for data_in in data['in_extraction']], [data_out for data_out in data['out_extraction']]
    list_in_complex_adjust = func_calculate_list_in_volterra_for_complex_number(list_in_complex, P, M)
    COEFS = func_find_coef_volterra(list_in_complex_adjust, lista_out_complex)
    list_in_complex_validation, lista_out_complex_validation = [data_in for data_in in data['in_validation']], [data_out for data_out in data['out_validation']]
    list_in_complex_adjust_validation = func_calculate_list_in_volterra_for_complex_number(list_in_complex_validation, P, M)
    y_estimate = func_calculate_y_estimate(list_in_complex_adjust_validation, COEFS)
    NMSE = func_calculate_NMSE_vetorizado(y_estimate, lista_out_complex_validation)
    func_plot_in_vs_out_amp(y_estimate, lista_out_complex_validation, list_in_complex)


def func_calculate_list_in_volterra_for_complex_number(list_in, P, M):
    """
    Está função ajusta a lista de entrada conforme descrito na equação.
    Args:
        list_in → Dados de entrada. (Array numpy de complex numbers)
        P → Grau do polinômio. (Inteiro)
        M → Memória do polinômio. (Inteiro)
    Return:
        list_in_volterra → Entrada adequada para cálculo do modelo.
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
                list_in_memo.append((abs(in_volterra)**(2*p-2))*in_volterra)
        list_in_volterra.append(list_in_memo)
    return list_in_volterra


def func_calculate_y_estimate(list_in_complex_adjust, COEFS):
    """
    Função que calcula a saída estimada a partir dos coeficientes e da entrada ajustada.
    Args:
        list_in_complex_adjust → Lista de entradas ajustadas. (List)
        COEFS → Coeficientes estimados. (np.ndarray)
    Returns:
        list_out_estimate → Lista de saídas estimadas. (List)
    """
    list_out_estimate = []
    for in_index in range(len(list_in_complex_adjust)):
        y_estimate = 0
        for m, in_volterra in enumerate(list_in_complex_adjust[in_index]):
            y_estimate += COEFS[m]*in_volterra
        list_out_estimate.append(y_estimate)
    return list_out_estimate


def func_calculate_NMSE_vetorizado(list_out_real, list_out_estimate, eps=1e-12):
    """
    Função que calcula o NMSE entre os valores reais e os estimados. (VERSÃO VETORIZADA)
    Args:
        list_out_real → Lista de saídas reais. (List)
        list_out_estimate → Lista de saídas estimadas. (List)
    Returns:
        nmse → Valor do NMSE. (Float)
   """
    list_out_real = np.asarray(list_out_real)
    list_out_estimate = np.asarray(list_out_estimate)
    num = np.sum(np.abs(list_out_real - list_out_estimate) ** 2)
    den = np.sum(np.abs(list_out_real) ** 2)
    nmse = 10 * np.log10(num / (den + eps))
    return float(nmse)


def func_calculate_NMSE(list_out_real, list_out_estimate):
    """
    Função que calcula o NMSE entre os valores reais e os estimados.
    Args:
        list_out_real → Lista de saídas reais. (List)
        list_out_estimate → Lista de saídas estimadas. (List)
    Returns:
        nmse → Valor do NMSE. (Float)
    """
    sum_error_numerator = 0
    sum_denominator = 0
    for i in range(len(list_out_estimate)):
        sum_error_numerator += (abs(list_out_real[i] - list_out_estimate[i]))**2
        sum_denominator = abs(list_out_real[i])**2
    nmse = 10*log10(sum_error_numerator/sum_denominator)
    return nmse


def func_plot_in_vs_out_amp(y_estimate, lista_out_complex_validation, list_in_complex, plot_type='AM-AM'):
    """
    Função que plota o gráfico de saída real versus saída estimada.
    Args:
        list_out_real → Lista de saídas reais. (List)
        list_out_estimate → Lista de saídas estimadas. (List)
    """
    plt.figure(figsize=(10, 6))
    list_real_amp = [abs(out_val[0]) for out_val in lista_out_complex_validation]
    list_estimate_amp = [abs(out_val[0]) for out_val in y_estimate]

    # gráfico de dispersão (pontos)
    plt.scatter(list_real_amp, list_estimate_amp, c='g', marker='o', label='Estimado vs Real')

    # linha de referência y=x (para comparar estimado ≈ real)
    min_val = min(min(list_real_amp), min(list_estimate_amp))
    max_val = max(max(list_real_amp), max(list_estimate_amp))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

    plt.xlabel('Entrada')
    plt.ylabel('Saída')
    plt.title('MODELO VOLTERRA COM COMPLEXOS')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
