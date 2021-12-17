from typing import List
import matplotlib.pyplot as plt
import torch
import wandb
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os.path
"""
    each dict should contain N-dimensional vectors under the following keys:
    'x_train', 'y_train',
    'x_test', 'y_test',
    'y_pred_mean', 'y_pred_std'
"""
def plot_predictions(plotting_data: List[dict], config: dict):
    fig = make_subplots(rows=2, cols=2)

    for i, data in enumerate(plotting_data):

        fig.append_trace(
            go.Scatter(
                x=data['x_test'],
                y=data['y_test'],
                line=dict(color='black', width=1),
                showlegend=False,
            ),
            row=(i % 2 + 1), col=(i // 2 + 1),
        )

        fig.append_trace(
            go.Scatter(
                x=data['x_train'],
                y=data['y_train'],
                mode='markers',
                marker=dict(size=15, color='red', symbol='cross'),
                showlegend=False,
            ),
            row=(i % 2 + 1), col=(i // 2 + 1),
        )

        fig.append_trace(
            go.Scatter(
                x=data['x_test'],
                y=data['y_pred_mean'],
                mode='lines',
                line=dict(color='green', width=2),
                showlegend=False,
            ),
            row=(i % 2 + 1), col=(i // 2 + 1),
        )

        if not all(data['y_pred_std'] == 0):
            fig.append_trace(
                go.Scatter(
                    name="Upper Bound",
                    x=data['x_test'],
                    y=data['y_pred_mean'] + data['y_pred_std'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                ),
                row=i % 2 + 1,
                col=i // 2 + 1,
            )

            fig.append_trace(
                go.Scatter(
                    name="Lower Bound",
                    x=data['x_test'],
                    y=data['y_pred_mean'] - data['y_pred_std'],
                    line=dict(width=0),
                    mode='lines',
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=False,
                ),
                row=i % 2 + 1,
                col=i // 2 + 1,
            )

    if config['wandb']:
        fig.write_image(os.path.join(config['logdir'], "test_predictions.png"))
        wandb.log({"test_predictions": wandb.Image(os.path.join(config['logdir'], "test_predictions.png"))})

    else:
        fig.show()
