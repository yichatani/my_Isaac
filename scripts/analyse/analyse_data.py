import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

# check_data = np.array([
# # 0.00339951, 0.02854688, 0.05301642, 0.07696556, 0.10031897, 0.12301677,
# #  0.14505669, 0.16644146, 0.18717751, 0.20726725, 0.22669552, 0.24546698,
# #  0.26367718, 0.28139395, 0.298612, 0.31534141, 0.33159074, 0.34741995,
# #  0.36284891, 0.37788874, 0.39254877, 0.40683988, 0.42077079, 0.43435004,
# #  0.44758925, 0.46049953, 0.47309208, 0.48537824, 0.49736843, 0.50907326,
# #  0.52049941, 0.53165048, 0.54253656, 0.55316758, 0.5635516, 0.57369858,
# #  0.58361673, 0.59331352, 0.60279757, 0.61207581, 0.62115622, 0.63004601,
# #  0.63875037, 0.64727777, 0.65563339, 0.66382319, 0.67185318, 0.67972893,
# #  0.68745542, 0.69503844, 
#  0.70248246, 0.70979208, 
# #  0.70979208+1e-3, 0.70979208+2e-3, 0.70979208+3e-3, 0.70979208+4e-3, 0.70979208+5e-3,
#  ])



data_str = """
            0.00288904 0.02554181 0.04742513 0.06869714 0.08931797 0.1092275
 0.12843339 0.14694659 0.16477957 0.18194705 0.19844542 0.21428555
 0.22956397 0.24433157 0.25859919 0.27237734 0.28567839 0.29852396
 0.31096733 0.32302085 0.33468908 0.3459844  0.35691929 0.36750704
 0.37775928 0.38768771 0.39730433 0.40662009 0.41564122 0.42437837
 0.43284309 0.44104457 0.44899303 0.45669782 0.4641678  0.47141132
 0.47843701 0.48525298 0.49186656 0.49828458 0.50451058 0.51055086
 0.51641464 0.52210528 0.52763003 0.53299308 0.53819948 0.54325539
 0.54816598 0.55293643 0.55757147 0.56207645 0.56645513 0.57071161
 0.57485062 0.57887542 0.58279061 0.58659971 0.5903061  0.59391218
 0.59742212 0.6008392  0.60416549 0.60740459 0.61055964 0.61363304
 0.61662591 0.61954296 0.62238574 0.62515581 0.62785548 0.63048768
 0.63305402 0.63555622 0.63799644 0.64037526 0.6426965  0.64496171
 0.64717108 0.64932692 0.65143037 0.65348351 0.65548772 0.65744323
 0.65935296 0.66121751 0.66303766 0.66481525 0.66655177 0.66824675
 0.66990262 0.67152047 0.67310137 0.67464548 0.67615438 0.67762899
 0.67906982 0.68047744 0.68185407 0.68319952 0.68451488 0.68580025
 0.68705773 0.68828678 0.68948835 0.69066274 0.69181156 0.69293475
 0.69403374 0.69510865 0.69615996 0.6971885  0.69819456 0.69917911
 0.70014191 0.7010839  0.70200574 0.70290744 0.70379001 0.7046541
             """
# data_str = """
#      0.64932692 0.65143037 0.65348351 0.65548772

# """

data_list = [float(x) for x in data_str.split()]
check_data = np.array(data_list)

def calculate_slope(x, y):
    # Fit a linear regression model to the data
    slope, intercept = np.polyfit(x, y, 1)
    return slope


def draw_width_vs_time():
    x_values = np.arange(len(check_data))
    print(len(x_values))
    y_values = check_data

    slope = calculate_slope(x_values, y_values)
    print(f"Slope: {slope}")

    print(len(y_values))
    plt.title("Width vs Time")
    plt.ylabel("Width")
    plt.xlabel("Time")
    plt.plot(x_values, y_values)
    plt.show()

def draw_slop_vs_width():
    window_size = 2
    half_win = window_size // 2
    x_values = np.arange(len(check_data))
    y_values = check_data

    slopes = []
    widths = []

    for i in range(half_win, len(y_values) - half_win):
        x_win = x_values[i - half_win:i + half_win]
        y_win = y_values[i - half_win:i + half_win]
        slope, _ = np.polyfit(x_win, y_win, 1)
        slopes.append(slope)
        widths.append(y_values[i])
    
    fit_coeffs = np.polyfit(widths, slopes, 1)
    a, b = fit_coeffs
    print(f"slope = {a:.6f} * width + {b:.6f}")
    fit_slopes = a * np.array(widths) + b

    plt.figure(figsize=(8, 5))
    plt.plot(widths, slopes, marker='o', linestyle='-')
    # plt.title("Slope vs Width")
    # plt.xlabel("Width")
    # plt.ylabel("Local Slope")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    plt.plot(widths, fit_slopes, '-', label=f'Fit: slope = {a:.4f} * width + {b:.4f}')
    plt.xlabel("Width")
    plt.ylabel("Slope")
    plt.title("Linear Fit: Slope vs Width")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # draw_width_vs_time()
    draw_slop_vs_width()
