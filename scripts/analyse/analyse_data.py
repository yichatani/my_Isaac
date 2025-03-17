import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

check_data = np.array([
# 0.00339951, 0.02854688, 0.05301642, 0.07696556, 0.10031897, 0.12301677,
#  0.14505669, 0.16644146, 0.18717751, 0.20726725, 0.22669552, 0.24546698,
#  0.26367718, 0.28139395, 0.298612, 0.31534141, 0.33159074, 0.34741995,
#  0.36284891, 0.37788874, 0.39254877, 0.40683988, 0.42077079, 0.43435004,
#  0.44758925, 0.46049953, 0.47309208, 0.48537824, 0.49736843, 0.50907326,
#  0.52049941, 0.53165048, 0.54253656, 0.55316758, 0.5635516, 0.57369858,
#  0.58361673, 0.59331352, 0.60279757, 0.61207581, 0.62115622, 0.63004601,
#  0.63875037, 0.64727777, 0.65563339, 0.66382319, 0.67185318, 0.67972893,
#  0.68745542, 0.69503844, 
 0.70248246, 0.70979208, 
#  0.70979208+1e-3, 0.70979208+2e-3, 0.70979208+3e-3, 0.70979208+4e-3, 0.70979208+5e-3,
 ])

def calculate_slope(x, y):
    # Fit a linear regression model to the data
    slope, intercept = np.polyfit(x, y, 1)
    return slope



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
