import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

import plotly as py
import plotly.graph_objs as go

from dash import dcc
from dash import html
from jupyter_dash import JupyterDash

py.offline.init_notebook_mode(connected=True)

def multiplot(lines, rows, off, data, title, plot_f,figsize=(20, 15) ,**kwargs):
    plt.figure(figsize=figsize)
    for i in np.arange(lines*rows - off):
        
        plt.subplot(lines, rows, i+1)
        plot_f(data[i], **kwargs)
        plt.title(title[i])
        
    plt.show()

def plot_forcast(obs_temperature=None,
                     pred_house_temperature=None,
                     pred_obs_temperature=None,
                     time=None,
                     TS=None,
                     title_t=None):

    median_predicted = np.percentile(pred_house_temperature, 50, axis=0)
    lower_predicted = np.percentile(pred_house_temperature,  15, axis=0)
    upper_predicted = np.percentile(pred_house_temperature, 85, axis=0)

    median_measure = np.percentile(pred_obs_temperature, 50, axis=0)
    lower_measure = np.percentile(pred_obs_temperature,  15, axis=0)
    upper_measure = np.percentile(pred_obs_temperature, 85, axis=0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x = np.concatenate([time.values, np.flip(time.values)]),
        y = np.concatenate([upper_measure, np.flip(lower_measure)]),
        fill='toself', # fill area between trace0 and trace1
        # mode='dot', 
        hoveron='points',
        line_color=' mediumslateblue',
        name="Posterior predictive - uncertainty"
        ))
    fig.add_trace(go.Scatter(
        x = np.concatenate([time.values, np.flip(time.values)]),
        y = np.concatenate([upper_predicted, np.flip(lower_predicted)]),
        fill='toself', # fill area between trace0 and trace1
        # mode='dot', 
        hoveron='points',
        line_color='lavender',
        name="Temperature (estimates) - uncertainty"
        ))

    fig.add_trace(go.Scatter(x=time.values, y=median_measure, fill=None, mode='lines', line_dash='dot', line_color=" mediumslateblue", name="Posterior predictive"))
    fig.add_trace(go.Scatter(x=time.values, y=median_predicted, fill=None, mode='lines', line_dash="dash", line_color="lavender", name="Temperature (estimates)"))
    fig.add_trace(go.Scatter(x=time.values, y=obs_temperature, fill=None, mode='lines', line_color="darkorange", name="Temperatures (measures)"))
    fig.add_trace(go.Scatter(x=time.values, y=TS, fill=None, mode='lines', line_color="cyan", name="Temperatures set"))

    fig.update_xaxes(title_text='Time', showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(title_text='°C', showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_layout(legend_title_text='Temperature curves', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="mintcream", bordercolor="Black", borderwidth=1), title={'y':0.93, 'x':0.18, 'yanchor':'top', "text":title_t}, title_font_size = 25, width=1350)
                    
    fig.show()

def interactive_plot(rooms, room, y_obs, y_pred, time_array, settemp_array):
    df = pd.DataFrame({'y_Obs': y_obs[room], 'y_Pred': y_pred[room]})
    new_df = df
    
    trace1 = go.Scatter(x = time_array,
                        y = settemp_array,
                        mode = 'lines',
                        name = 'Set Temperature',
                        line = dict(shape='spline'))
                            
    trace2 = go.Scatter(x = time_array,
                        y = new_df['y_Obs'],
                        mode = 'lines',
                        name = 'Test Output',
                        line = dict(shape='spline'))

    trace3 = go.Scatter(x = time_array,
                        y = new_df['y_Pred'],
                        mode = 'lines',
                        name = 'Predictive Output',
                        line = dict(shape='spline'))

    Zones = ['Dining room', 'Kitchen', 'Living Room', 'Bedroom 1', 'Bathroom', 'Bedroom 2', 'Bedroom 3']
    layout = go.Layout(title = Zones[room], xaxis = dict(title = 'Time'), yaxis = dict(title = '°C'))

    data=[trace1, trace2, trace3]
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