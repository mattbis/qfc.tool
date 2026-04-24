import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re

# ===== GLOBAL STATE =====
params = {}
sliders = {}

# ===== SAFE EVAL =====
def safe_eval(expr, x, params):
    allowed = {
        "x": x,
        "pi": np.pi,
        "sin": np.sin,
        "cos": np.cos,
        "np": np
    }
    allowed.update(params)
    return eval(expr, {"__builtins__": {}}, allowed)

# ===== FIND VARIABLES =====
def extract_params(expr):
    tokens = set(re.findall(r"[a-zA-Z_]\w*", expr))
    blacklist = {"x", "sin", "cos", "pi", "np"}
    return sorted(list(tokens - blacklist))

# ===== UPDATE PLOT =====
def update_plot():
    try:
        expr = formula_entry.get()

        # update param values
        for k in sliders:
            params[k] = sliders[k].get()

        xs = np.linspace(0, 1, 20000)
        ys = safe_eval(expr, xs, params)

        y_min = float(np.min(ys))
        y_max = float(np.max(ys))

        ax.clear()
        ax.plot(xs, ys)
        ax.set_title(f"min={y_min:.4f}  max={y_max:.4f}")
        canvas.draw()

    except Exception as e:
        ax.clear()
        ax.set_title(f"Error: {str(e)}")
        canvas.draw()

# ===== BUILD SLIDERS =====
def build_sliders():
    global sliders
    for widget in slider_frame.winfo_children():
        widget.destroy()

    sliders = {}
    expr = formula_entry.get()
    found = extract_params(expr)

    for name in found:
        params[name] = 1.0

        frame = ttk.Frame(slider_frame)
        frame.pack(fill="x")

        label = ttk.Label(frame, text=name, width=8)
        label.pack(side="left")

        slider = tk.Scale(frame, from_=-10, to=10,
                          resolution=0.01,
                          orient="horizontal",
                          command=lambda e: update_plot())
        slider.set(1.0)
        slider.pack(fill="x", expand=True)

        sliders[name] = slider

    update_plot()

# ===== UI =====
root = tk.Tk()
root.title("DSP Formula Lab")

top_frame = ttk.Frame(root)
top_frame.pack(fill="x")

formula_entry = ttk.Entry(top_frame)
formula_entry.pack(fill="x", padx=5, pady=5)
formula_entry.insert(0, "0.7-(((2-(((cos(x*pi*c1))*cos(x*pi*c2)*amp)*mul))*scale)/norm)*(sin(x*freq/pi))")

btn = ttk.Button(top_frame, text="Build Sliders", command=build_sliders)
btn.pack(pady=5)

slider_frame = ttk.Frame(root)
slider_frame.pack(fill="both", expand=True)

# ===== PLOT =====
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill="both", expand=True)

# ===== START =====
build_sliders()
root.mainloop()
