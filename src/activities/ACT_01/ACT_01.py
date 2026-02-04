from src.activities.LIB_ML import f_expr_somat_symb, f_find_coef_analytically, f_find_coef_numerically, f_calculate_y_estimate, f_plot_xy

def main():

    # Lists defined as test data.
    list_x_real, list_y_real = [0.0, 0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.45, 0.7, 0.8]

    # The function f_expr_somat_symb returns the analytical expression we need to differentiate with respect to
    # w and b. To differentiate, we use the resources of the sympy library, which solves calculus problems. Now we
    # calculate the symbolic expression using the developed function.
    expression = f_expr_somat_symb(list_x_real, list_y_real)

    # So, using the function f_expr_somat_symb, we arrive at 5.0b^2 + 2.0bw - 4.9b + 0.3w^2 - 1.3w + 1.4625.
    # Differentiating them with respect to w and b and finding their roots, we have.
    w, b = f_find_coef_analytically(expression)

    # Now, we want to plot the estimated output versus the actual output. To do this, we need to calculate list_y_est.
    list_y_est = f_calculate_y_estimate(list_x_real, w, b)

    # Now, we can plot.
    f_plot_xy([list_x_real, list_x_real], [list_y_real, list_y_est], ['Real', 'Estimate'])

    # In Using optimized libraries that calculate the coefficients numerically, and plot.
    w, b = f_find_coef_numerically(list_x_real, list_y_real)
    list_y_est = f_calculate_y_estimate(list_x_real, w, b)
    f_plot_xy([list_x_real, list_x_real], [list_y_real, list_y_est], ['Real', 'Estimate'])

if __name__ == "__main__":
    main()