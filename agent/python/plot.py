import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

import plotly as py
import plotly.graph_objs as go

import seaborn as sns
sns.set_theme()

py.offline.init_notebook_mode(connected=True)

def interactive_plot(title, time_array, traces:dict):
    data = []

    for name, values in traces.items():
        data.append(go.Scatter(x = time_array,
                        y = values,
                        mode = 'lines',
                        name = name,
                        line = dict(shape='spline')))

    layout = go.Layout(title = title, xaxis = dict(title = 'Time'), yaxis = dict(title = '°C'))

    fig = go.Figure(data, layout=layout)

    fig.update_xaxes(title_text='Time', showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(title_text='°C', showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_layout(legend_title_text='Temperature curves', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="mintcream", bordercolor="Black", borderwidth=1), title={'y':0.875, 'x':0.18, 'yanchor':'top'}, title_font_size = 25)

    fig.update_layout(
        autosize=False,
        width=1300,
    )

    fig.show()

def multi_plot(datetime,title, Ts, Tz, Q_heat, Tw, Tr, switch, Ta=None):
    x = datetime
    # plt.figure(figsize=(20,10))   
    plt.close()

    fig, axs = plt.subplots(3, sharex=True, sharey=False)
    axs[0].plot(x, Ts, label="Temperature set", linewidth=2)
    axs[0].plot(x, Tz, label="Temperature of the zone", linewidth=2)
    if Ta is not None:
        axs[0].plot(x, Ta, ".",label="Outside temperature", linewidth=1 )
    axs[0].fill_between(x, Ts, Tz, where=Tz >= Ts, facecolor='orange', interpolate=True, alpha=0.2)
    axs[0].fill_between(x, Ts, Tz, where=Tz <= Ts, facecolor='red', interpolate=True, alpha=0.2)
    ax2 = axs[0].twinx()
    color = 'tab:red'
    ax2.set_ylabel('Switch', color=color)  # we already handled the label with ax1
    ax2.plot(x, switch, "--", color=color, alpha=0.5)
    # ax2.tick_params(axis='y', labelcolor=color)
    axs[0].legend(loc="upper left", fontsize="large")
    axs[0].xaxis_date()
    axs[0].set_ylabel("°C")
    axs[0].tick_params(left = False, right=False)
    fig.suptitle(title, fontsize=30)
    ax2.tick_params(left = False, right=False)
    axs[1].plot(x, Q_heat, label="Q_heat", linewidth=2)
    axs[1].legend(loc="upper left", fontsize="large")
    axs[1].xaxis_date()
    axs[1].set_ylabel("W")
    axs[2].plot(x, Tw, label="Temperature water", linewidth=2)
    axs[2].plot(x, Tr, "--", label="Temperature radiator", linewidth=2)
    axs[2].legend(loc="upper left", fontsize="large")
    axs[2].set_ylabel("°C")
    fig.autofmt_xdate()

    fig.set_figheight(15)
    fig.set_figwidth(20)
    plt.show()
