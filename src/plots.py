import matplotlib.pyplot as plt
import numpy as np


def plot(algorithm):
    f = open(r'../stats/' + algorithm, 'r')
    row = f.readline()
    if algorithm == 'Nearest neighbours':
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(14, 5)
        fig.canvas.set_window_title(algorithm)
        y = [float(i) for i in row.split()]
        x = ['2', '1', 'Inf', 'Cos']
        ax1.plot(x, y)
        ax1.set(xlabel='Norm', ylabel='Precision (%)',
                title='Nearest neighbours: Precision vs Norm')
        ax1.grid()
        row = f.readline()
        y2 = [float(i) for i in row.split()]
        x2 = ['2', '1', 'Inf', 'Cos']
        ax2.plot(x2, y2, color='orange')
        ax2.set(xlabel='Norm', ylabel='Avg Query Time (ms)',
                title='Nearest neighbours: AQT vs Norm')
        ax2.grid()
        plt.show()
    elif algorithm == 'k-Nearest neighbours':
        fig, ax = plt.subplots()
        fig.set_size_inches(7, 5)
        fig.canvas.set_window_title(algorithm)
        x = [float(i) for i in row.split()]
        norms = ['2', '1', 'Inf', 'Cos']
        for norm in norms:
            row = f.readline()
            precisions = [float(i) for i in row.split()]
            ax.plot(x, precisions, label=norm)
        ax.set(xlabel='k (#)', ylabel='Precision (%)',
                title=algorithm + ': RR(k)')
        ax.grid()
        ax.legend()
        plt.show()
    elif algorithm == 'Tensori A1':
        fig, ax = plt.subplots()
        fig.set_size_inches(7, 5)
        fig.canvas.set_window_title(algorithm)
        ppt = row
        row = f.readline()
        y = [float(i) for i in row.split()]
        x = ['2', '1', 'Inf', 'Cos']
        ax.plot(x, y)
        ax.set(xlabel='Norm', ylabel='Precision (%)',
                title='Tensori A1: Precision vs Norm')
        ax.grid()
        plt.show()
    else:
        print(algorithm)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(14, 5)
        fig.canvas.set_window_title(algorithm)
        x = [float(i) for i in row.split()]
        norms = ['2', '1', 'Inf', 'Cos']
        row = f.readline()
        ppt = [float(i) for i in row.split()]
        for norm in norms:
            row = f.readline()
            precisions = [float(i) for i in row.split()]
            ax1.plot(x, precisions, label=norm)
        ax1.set(xlabel='k (#)', ylabel='Precision (%)',
                title=algorithm + ': RR(k)')
        ax1.grid()
        ax1.legend()

        ax2.plot(x, ppt, color='orange')
        ax2.set(xlabel='k (#)', ylabel='Pre-processing time (ms)',
                title=algorithm + ': P-PTime(k)')
        ax2.grid()
        plt.show()