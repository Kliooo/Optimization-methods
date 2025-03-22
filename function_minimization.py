import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from scipy.optimize import minimize_scalar


def f1(x):
    x1, x2 = x
    return 100 * (x2 - x1**2)**2 + 5 * (1 - x1)**2

def f2(x):
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2


def nelder_method(func, start_point, alpha=1, beta=0.5, gamma=2, epsilon=1e-8, max_iter=1000):
    n = len(start_point)

    # Формируем начальный симплекс
    simplex = np.array([start_point])
    for i in range(n):
        x_new = start_point.copy()
        x_new[i] += 0.5
        simplex = np.vstack([simplex, x_new])

    func_vals = np.array([func(x) for x in simplex])

    iter_count = 0
    while iter_count < max_iter:
        sorted_indices = np.argsort(func_vals)
        simplex = simplex[sorted_indices]
        func_vals = func_vals[sorted_indices]
        
        centroid = np.mean(simplex[:-1], axis=0)

        reflection = centroid + alpha * (centroid - simplex[-1])
        reflection_val = func(reflection)

        if func_vals[0] <= reflection_val < func_vals[-2]:
            simplex[-1] = reflection
            func_vals[-1] = reflection_val
        elif reflection_val < func_vals[0]:
            expansion = centroid + gamma * (reflection - centroid)
            expansion_val = func(expansion)
            if expansion_val < reflection_val:
                simplex[-1] = expansion
                func_vals[-1] = expansion_val
            else:
                simplex[-1] = reflection
                func_vals[-1] = reflection_val
        else:
            contraction = centroid + beta * (simplex[-1] - centroid)
            contraction_val = func(contraction)
            if contraction_val < func_vals[-1]:
                simplex[-1] = contraction
                func_vals[-1] = contraction_val
            else:
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                func_vals[1:] = np.array([func(x) for x in simplex[1:]])

        if np.max(np.abs(simplex - centroid)) < epsilon:
            break

        iter_count += 1

    min_point = simplex[0]
    min_value = func_vals[0]
    
    return min_point, min_value, iter_count

def gradient(func, x, epsilon=1e-8):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        def func_i(xi):
            x_copy = x.copy()
            x_copy[i] = xi
            return func(x_copy)
        
        grad[i] = (func_i(x[i] + epsilon) - func_i(x[i] - epsilon)) / (2 * epsilon)
    
    return grad

def gradient_method(func, start_point, epsilon=1e-8, max_iter=10000):
    x_k = np.array(start_point)
    iter_count = 0

    while iter_count < max_iter:
        grad_k = gradient(func, x_k, epsilon)
        
        if np.linalg.norm(grad_k) < epsilon:
            break

        def line_search(alpha):
            return func(x_k - alpha * grad_k)

        result = minimize_scalar(line_search)
        alpha_k = result.x
        x_k = x_k - alpha_k * grad_k
        iter_count += 1

    return x_k, func(x_k), iter_count

def conjugate_gradient_method(func, start_point, epsilon=1e-8, max_iter=10000):
    x_k = np.array(start_point)
    grad_k = gradient(func, x_k, epsilon)
    d_k = -grad_k
    iter_count = 0

    while iter_count < max_iter:
        if np.linalg.norm(grad_k) < epsilon:
            break

        def line_search(alpha):
            return func(x_k - alpha * d_k)

        result = minimize_scalar(line_search)
        alpha_k = result.x

        x_k_new = x_k - alpha_k * d_k

        grad_k_new = gradient(func, x_k_new, epsilon)

        beta_k = np.dot(grad_k_new, grad_k_new) / np.dot(grad_k, grad_k)
        d_k = -grad_k_new + beta_k * d_k

        x_k = x_k_new
        grad_k = grad_k_new
        iter_count += 1

    return x_k, func(x_k), iter_count

#Гессиан - матрица вторых производных функции в текущей точке, которая показывает кривизну функции
def hessian(func, x, epsilon=1e-5):
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_ij1, x_ij2, x_ij3, x_ij4 = [x.copy() for _ in range(4)]
            x_ij1[i] += epsilon
            x_ij1[j] += epsilon
            x_ij2[i] += epsilon
            x_ij2[j] -= epsilon
            x_ij3[i] -= epsilon
            x_ij3[j] += epsilon
            x_ij4[i] -= epsilon
            x_ij4[j] -= epsilon
            
            hess[i, j] = (func(x_ij1) - func(x_ij2) - func(x_ij3) + func(x_ij4)) / (4 * epsilon**2)
    return hess

def newtonian_method(func, start_point, epsilon=1e-8, max_iter=1000, lambda_reg=1e-2):
    x_k = np.array(start_point, dtype=float)
    iter_count = 0
    
    while iter_count < max_iter:
        grad_k = gradient(func, x_k, epsilon)
        
        if np.linalg.norm(grad_k) < epsilon:
            break
        
        hess_k = hessian(func, x_k, epsilon)
        
        while True:
            try:
                hess_reg = hess_k + lambda_reg * np.eye(len(x_k))
                p_k = -np.linalg.solve(hess_reg, grad_k)
                break
            except np.linalg.LinAlgError:
                lambda_reg *= 10
        
        def line_search(alpha):
            return func(x_k + alpha * p_k)
        
        result = minimize_scalar(line_search)
        alpha_k = result.x if result.success else 1.0
        
        x_k += alpha_k * p_k
        iter_count += 1
    
    return x_k, func(x_k), iter_count

def update_plot():
    selected_func = func_combo.get()
    selected_method = method_combo.get()
    
    if selected_func == "f1(x) = 100(x2 - x1^2)^2 + 5(1 - x1)^2":
        func = f1
    else:
        func = f2
    
    try:
        x1_st = x1_start.get().strip()
        x2_st = x2_start.get().strip()

        if not x1_st or not x2_st:
            raise ValueError("Пустое поле ввода")
        
        x1_st = float(x1_st)
        x2_st = float(x2_st)
        start_point = [x1_st, x2_st]
        
    except ValueError as e:
        print(f"Ошибка: Неверный ввод для стартовой точки ({e})")
        return
    
    if selected_method == "Метод деформируемого многогранника":
        min_point, min_value, iterations = nelder_method(func, start_point)
    elif selected_method == "Градиентный метод":
        min_point, min_value, iterations = gradient_method(func, start_point)
    elif selected_method == "Метод сопряженных градиентов":
        min_point, min_value, iterations = conjugate_gradient_method(func, start_point)
    elif selected_method == "Ньютоновский метод":
        min_point, min_value, iterations = newtonian_method(func, start_point)
    
    min_value_entry.delete(0, tk.END)
    min_value_entry.insert(0, f"{min_value:.8f}")

    x1_entry.delete(0, tk.END)
    x1_entry.insert(0, f"{min_point[0]:.8f}")

    x2_entry.delete(0, tk.END)
    x2_entry.insert(0, f"{min_point[1]:.8f}")
    
    iteration_entry.delete(0, tk.END)
    iteration_entry.insert(0, str(iterations))

    ax.clear()
    ax.set_title("График функции", pad=40)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1, x2)")

    x1_vals = np.linspace(min_point[0] - 3, min_point[0] + 3, 100)
    x2_vals = np.linspace(min_point[1] - 3, min_point[1] + 3, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    
    Z = np.zeros(X1.shape)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = func([X1[i, j], X2[i, j]])
    
    ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none', alpha=0.7)
    
    ax.scatter(min_point[0], min_point[1], min_value, color='r', s=150, edgecolor='k', label="Минимум")
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    ax.view_init(elev=30, azim=30)
    
    canvas.draw()


root = tk.Tk()
root.title("Минимизация функций")
root.geometry("1000x600")

left_frame = tk.Frame(root, padx=10, pady=10, width=430)
left_frame.pack(side="left", fill="y", anchor="s")
left_frame.pack_propagate(False)

params_label = tk.LabelFrame(left_frame, text="Параметры", padx=10, pady=10)
params_label.pack(fill="x", pady=5)
params_label.grid_columnconfigure(1, weight=1)

tk.Label(params_label, text="Функция:").grid(row=0, column=0, sticky="w", pady=5)
func_combo = ttk.Combobox(params_label, values=["f1(x) = 100(x2 - x1^2)^2 + 5(1 - x1)^2", "f2(x) = (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2"], state="readonly")
func_combo.grid(row=0, column=1, sticky="ew", pady=5, padx=(5, 0))
func_combo.current(0)

tk.Label(params_label, text="Метод:").grid(row=1, column=0, sticky="w", pady=5)
method_combo = ttk.Combobox(params_label, values=["Метод деформируемого многогранника", "Градиентный метод", "Метод сопряженных градиентов", "Ньютоновский метод"], state="readonly")
method_combo.grid(row=1, column=1, sticky="ew", pady=5, padx=(5, 0))
method_combo.current(0)

tk.Label(params_label, text="Стартовая точка x1:").grid(row=2, column=0, sticky="w", pady=5)
x1_start = tk.Entry(params_label)
x1_start.grid(row=2, column=1, sticky="ew", pady=5, padx=(5, 0))
x1_start.insert(0, "0.0")

tk.Label(params_label, text="Стартовая точка x2:").grid(row=3, column=0, sticky="w", pady=5)
x2_start = tk.Entry(params_label)
x2_start.grid(row=3, column=1, sticky="ew", pady=5, padx=(5, 0))
x2_start.insert(0, "0.0")

calculate_btn = tk.Button(left_frame, text="Минимизировать", command=update_plot)
calculate_btn.pack(fill="x", pady=5)

results_label = tk.LabelFrame(left_frame, text="Результаты", padx=10, pady=10)
results_label.pack(side="bottom", fill="x", pady=5)
results_label.grid_columnconfigure(1, weight=1)

tk.Label(results_label, text="Значение функции:").grid(row=0, column=0, sticky="w", pady=5)
min_value_entry = tk.Entry(results_label)
min_value_entry.grid(row=0, column=1, sticky="ew", pady=5, padx=(5, 0))

tk.Label(results_label, text="Значение x1:").grid(row=1, column=0, sticky="w", pady=5)
x1_entry = tk.Entry(results_label)
x1_entry.grid(row=1, column=1, sticky="ew", pady=5, padx=(5, 0))

tk.Label(results_label, text="Значение x2:").grid(row=2, column=0, sticky="w", pady=5)
x2_entry = tk.Entry(results_label)
x2_entry.grid(row=2, column=1, sticky="ew", pady=5, padx=(5, 0))

tk.Label(results_label, text="Количество итераций:").grid(row=3, column=0, sticky="w", pady=5)
iteration_entry = tk.Entry(results_label)
iteration_entry.grid(row=3, column=1, sticky="ew", pady=5, padx=(5, 0))

right_frame = tk.Frame(root, padx=10, pady=10)
right_frame.pack(side="right", fill="both", expand=True)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas.get_tk_widget().pack(fill="both", expand=True)

root.mainloop()
