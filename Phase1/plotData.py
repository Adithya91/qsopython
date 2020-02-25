import matplotlib.pyplot as plt
import math 

#To generate the plot
def plot(x, y, col, xlab, ylab, main, sub):
    plt.plot(x, y, color=col)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.suptitle(main)
    plt.figtext(
        0.5,
        0.01,
        sub,
        wrap=True,
        horizontalalignment='center',
        fontsize=12)
    plt.legend()
    plt.show()


def gaussian(x,a,b,c):
 return (a * math.exp((-0.5) * ((x - b)*(x - b)) / (c*c)))
