from src.activities.defines import P, M, file_IN_OUT_PA
from src.activities.LIB_ML import f_calculate_array_in_volterra, f_find_coef_volterra, f_show_coefs
from scipy.io import loadmat
import sympy as sp
import numpy as np


def main():

    # Extraction of real data collected in the laboratory and adaptation to Python data type. Definition of P and M.
    data = loadmat(file_IN_OUT_PA)
    array_in, array_out = np.array([data_in for data_in in data['in']]), np.array([data_out for data_out in data['out']])

    # Calculation of the Volterra series model output using the function developed in LIB_ML.
    X = np.array(f_calculate_array_in_volterra(array_in, P, M))

    # Calculation of the Volterra series model coefficients using the function developed in LIB_ML.
    COEFS = f_find_coef_volterra(X, array_out)

    print(f"Calculated Coefficients: {f_show_coefs(COEFS)}")
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred:", e)