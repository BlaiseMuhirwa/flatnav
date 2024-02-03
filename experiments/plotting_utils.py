from copy import deepcopy
import plotly.express as px
import pandas as pd
import json
import matplotlib.pyplot as plt
import re

# def plot_bubble_chart(metrics: dict):
#     # Adding a placeholder for the size of the datasets
#     df["DataSize"] = pd.Series(
#         [1 for _ in range(len(df))]
#     )  # Replace with actual data size values

#     fig = px.scatter(
#         df,
#         x="Skewness",
#         y="Latency",
#         size="DataSize",  # This is the third variable encoded as bubble size
#         color="Algorithm",
#         hover_name="Dataset",
#     )
#     fig.show()


def plot_heatmap(metrics: dict):
    df_hnsw = pd.DataFrame(
        {
            "Skewness": metrics["skewness_hnsw"],
            "Latency": metrics["latency_hnsw"],
            "Algorithm": "HNSW",
            "Dataset": metrics["dataset_names"],
        }
    )
    df_flatnav = pd.DataFrame(
        {
            "Skewness": metrics["skewness_flatnav"],
            "Latency": metrics["latency_flatnav"],
            "Algorithm": "FlatNav",
            "Dataset": metrics["dataset_names"],
        }
    )
    df = pd.concat([df_hnsw, df_flatnav])

    # Pivot for heatmap
    heatmap_data = df.pivot_table(
        values="Latency", index="Dataset", columns="Algorithm", aggfunc="mean"
    )

    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Algorithm", y="Dataset", color="Latency"),
        x=["HNSW", "FlatNav"],
    )
    fig.show()
    
def plot_metrics_plotly(metrics: dict, k: int):
    df_hnsw = pd.DataFrame(
        {
            "Skewness": metrics["skewness_hnsw"],
            "Latency": metrics["latency_hnsw"],
            "Algorithm": "HNSW",
            "Dataset": metrics["dataset_names"],
        }
    )
    df_flatnav = pd.DataFrame(
        {
            "Skewness": metrics["skewness_flatnav"],
            "Latency": metrics["latency_flatnav"],
            "Algorithm": "FlatNav",
            "Dataset": metrics["dataset_names"],
        }
    )
    df = pd.concat([df_hnsw, df_flatnav], ignore_index=True)

    # Create the scatter plot
    fig = px.scatter(
        df,
        x="Skewness",
        y="Latency",
        color="Algorithm",
        symbol="Algorithm",
        size_max=15,
        hover_name="Dataset",  # Shows dataset name on hover
        title="Mean query latency vs hubness score",
    )
    fig.update_layout(
        legend_title_text="Algorithm",
        xaxis_title="Skewness",
        yaxis_title="Latency",
        legend=dict(orientation="h", yanchor="top", y=0.01, xanchor="left", x=0.01),
    )
    fig.show()
    fig.write_html("hubness__.html")

def plot_faceted_grid(metrics: dict):
    df_hnsw = pd.DataFrame(
        {
            "Skewness": metrics["skewness_hnsw"],
            "Latency": metrics["latency_hnsw"],
            "Algorithm": "HNSW",
            "Dataset": metrics["dataset_names"],
        }
    )
    df_flatnav = pd.DataFrame(
        {
            "Skewness": metrics["skewness_flatnav"],
            "Latency": metrics["latency_flatnav"],
            "Algorithm": "FlatNav",
            "Dataset": metrics["dataset_names"],
        }
    )
    df = pd.concat([df_hnsw, df_flatnav])

    # Create the faceted grid
    fig = px.scatter(
        df,
        x="Skewness",
        y="Latency",
        color="Algorithm",
        facet_col="Algorithm",
        hover_name="Dataset",
    )
    fig.show()
    

def plot_bubble_chart(metrics, algorithm_name, ax):
    """
    Plots a bubble chart from a metrics dictionary.

    Parameters:
    - metrics: A dictionary containing 'latency', 'skewness', and 'dataset_names' keys.
    - algorithm_name: The name of the algorithm to be used to extract latency and skewness.
    - ax: The matplotlib axis on which to plot the chart.
    """
    # Extract the latency and skewness values based on the algorithm name
    latency_key = f"latency_{algorithm_name.lower()}"
    skewness_key = f"skewness_{algorithm_name.lower()}"
    latencies = metrics[latency_key]
    skewnesses = metrics[skewness_key]
    names = metrics['dataset_names']
    
    # Extract dimensions from the dataset names
    dimensions = [int(re.search(r'\d+', name).group()) for name in names]
    
    # Normalize dimensions for bubble size: the smallest dimension gets a size of 100, and others are scaled proportionally
    min_dimension = min(dimensions)
    sizes = [(dim / min_dimension) * 100 for dim in dimensions]
    
    # Scatter plot
    scatter = ax.scatter(skewnesses, latencies, s=sizes, alpha=0.5, label=algorithm_name)
    
    # Label the axes
    ax.set_xlabel('Skewness')
    ax.set_ylabel('Latency')
    ax.set_title(f'Bubble Chart for {algorithm_name}')
    
    # Add a legend with a title
    legend = ax.legend(title="Dataset Dimension")
    
    # Update the sizes in the legend using the new attribute 'legend_handles'
    for handle, dim in zip(legend.legend_handles, sorted(set(dimensions))):
        handle.set_sizes([dim])

    return ax


    
    
if __name__=="__main__":
    # load metrics from JSON 
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    
    metrics2 = deepcopy(metrics)
    fig, ax = plt.subplots()
    ax = plot_bubble_chart(metrics, 'FlatNav', ax)
    ax = plot_bubble_chart(metrics2, 'HNSW', ax)
    
    # Save the plot
    plt.savefig("bubble_chart.png")
    
    plt.show()
        
    # plot the heatmap
    # plot_metrics_plotly(metrics, 100)        
    
