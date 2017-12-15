import numpy as np
import math
import matplotlib.pyplot as plt

def draw_line(axes, y_func, line_label, line_color):
    px = np.arange(1, 300, 1)
    py = y_func(px)
    axes.plot(px, py, color=line_color, label=line_label)

if __name__ == '__main__':
    fig = plt.figure(figsize=(14,4))

    ax1 = fig.add_subplot(131)
    ax1.set_xlabel("trial")
    ax1.set_ylabel("episilon")
    ax1.set_title("$\epsilon = a^t$")
    draw_line(ax1, lambda t: 0.2 ** t, 'a=0.2', 'r')
    draw_line(ax1, lambda t: 0.5 ** t, 'a=0.5', 'g')
    draw_line(ax1, lambda t: 0.8 ** t, 'a=0.8', 'b')
    draw_line(ax1, lambda t: 0.8 ** t, 'a=0.99', 'y')
    ax1.legend(loc='upper right')
    ax1.grid(True, axis='both')

    ax2 = fig.add_subplot(132)
    ax2.set_xlabel("trial")
    ax2.set_ylabel("episilon")
    ax2.set_title("$\epsilon = 1/{t^2}$")
    draw_line(ax2, lambda t: 1./t**2, '', 'r')
    ax2.grid(True, axis='both')    

    ax3 = fig.add_subplot(133)
    ax3.set_xlabel("trial")
    ax3.set_ylabel("episilon")
    ax3.set_title("$\epsilon = e^{-at}$")
    draw_line(ax3, lambda t: math.e ** -(0.9*t), 'a=0.9', 'r')
    draw_line(ax3, lambda t: math.e ** -(0.1*t), 'a=0.1', 'g')
    draw_line(ax3, lambda t: math.e ** -(0.05*t), 'a=0.05', 'b')
    draw_line(ax3, lambda t: math.e ** -(0.03*t), 'a=0.03', 'y')
    draw_line(ax3, lambda t: math.e ** -(0.01*t), 'a=0.01', 'black')
    ax3.legend(loc='upper right')
    ax3.grid(True, axis='both')    

    plt.show()