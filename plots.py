import numpy as np
import matplotlib
matplotlib.use('ps')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

plt.style.use('ggplot')

matplotlib.rc('text', usetex=True)
matplotlib.rc('xtick', labelsize=7)
matplotlib.rc('ytick', labelsize=7)
matplotlib.rc('font', size=3, family='serif')
matplotlib.rc('axes', titlesize=8, labelsize=8)
matplotlib.rc('legend', fontsize=8)


names = ['dp-linear', 'dp-monotone', 'dp-linear-hirsch', 'dp-monotone-hirsch', 'wilber']

colors = {
    'dp-linear': 'blue',
    'dp-monotone': 'red',
    'dp-linear-hirsch': 'navy',
    'dp-monotone-hirsch': 'magenta',
    'wilber': 'green',
}

markers = {
    'dp-linear': '+',
    'dp-monotone': 'x',
    'dp-linear-hirsch': 'o',
    'dp-monotone-hirsch': 'd',
    'wilber': '^',
}

extension = 'pdf'
dots_per_inch = 72.27
width = 469.75502 / dots_per_inch / 2
height = width

def make_legend():
    return

def show_fixed_k(data, k):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0.16, bottom=0.17, right=0.95, top=0.90)
    handles, labels = ax.get_legend_handles_labels()
    for i, name in enumerate(names):
        ax.plot(data.n.values,
                np.poly1d(np.polyfit(data.n.values, data[name].values/1e3, 1))(data['n'].values),
                '-', color=colors[name], label=(name + ' best fit'))
        ax.plot(data.n.values, data[name].values /1e3,
                marker=markers[name], linestyle='None', markerfacecolor='None',
                markeredgecolor=colors[name], markersize=2,
                label=name)
        line = mlines.Line2D([], [], marker=markers[name],
                             color=colors[name], markersize=2, label=name, markerfacecolor='None',
                             markeredgecolor=colors[name])
        handles.append(line)
        labels.append(name)
    ax.set_ylabel('time (seconds)')
    ax.set_xlabel('input size (n)')
    ax.set_title('K = %s' % k)
    ax.legend(handles, labels, loc='upper left')
    _, y_max = ax.set_ylim()
    ax.set_ylim(0, y_max)
    return fig

def show_fixed_n(data, n):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(1, 1, 1)
    handles, labels = ax.get_legend_handles_labels()
    fig.subplots_adjust(left=0.16, bottom=0.17, right=0.95, top=0.90)
    for i, name in enumerate(names):
        ax.plot(data.k.values,
                np.poly1d(np.polyfit(data.k.values, data[name].values/1e3, 1))(data['k'].values),
                '-', color=colors[name], label=(name + ' best fit'))
        ax.plot(data.k.values, data[name].values /1e3,
                marker=markers[name], linestyle='None', markerfacecolor='None',
                markeredgecolor=colors[name], markersize=2,
                label=name)
        line = mlines.Line2D([], [], marker=markers[name],
                             color=colors[name], markersize=2, label=name, markerfacecolor='None',
                             markeredgecolor=colors[name])
        handles.append(line)
        labels.append(name)
    ax.set_ylabel('time (seconds)')
    ax.set_xlabel('input size (k)')
    ax.set_title('N = %s' % n)
    ax.legend(handles, labels, loc='upper left')
    _, y_max = ax.set_ylim()
    ax.set_ylim(0, y_max)
    ax.set_xlim(8, 52)
    return fig


if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    k20 = df[df['k'] == 20]
    fig = show_fixed_k(k20, 20)
    fig.savefig('fixed_k.%s' % extension )
    n7 = df[df.n == 10000000]
    fig2 = show_fixed_n(n7, '10 000 000')
    fig2.savefig('fixed_n.%s' % extension)
