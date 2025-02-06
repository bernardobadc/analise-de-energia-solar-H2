import os
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

# Configuração de estilo para fundo escuro
plt.style.use("dark_background")

def export_figure(filename: str) -> None:
    """
    Saves the current figure as an image in the 'images' directory.
    
    Parameters
    ----------
    filename : str
        Name of the output image file.
    """
    output_dir = os.path.join(os.getcwd(), "images")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches="tight")

def plot_hourly_distribution(data: pd.DataFrame, save_image: bool = False, **kwargs) -> None:
    """
    Plots the hourly distribution of solar energy production.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data with datetime index and energy production values.
    save_image : bool, optional
        Whether to save the generated plot as an image, by default False.
    """
    energy_by_hour = data.groupby(data.index.hour).sum()
    
    plt.figure(figsize=(18, 6))
    energy_by_hour.plot(kind="bar", **kwargs)
    plt.title("Distribuição Horária da Energia Solar")
    plt.xlabel("Hora do Dia")
    plt.ylabel("Energia Solar (kWh)")
    plt.xticks(rotation=0)
    
    if save_image:
        export_figure(filename="distribuicao_horaria.png")
    plt.show()

def plot_top_years(data: pd.DataFrame, top_n: int = 3, save_image: bool = False, **kwargs) -> None:
    """
    Plots the top N years with the highest solar energy production.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data with datetime index and energy production values.
    top_n : int, optional
        Number of top years to display, by default 3.
    save_image : bool, optional
        Whether to save the generated plot as an image, by default False.
    """
    energy_by_year = data.groupby(data.index.year).sum()
    top_years = energy_by_year.nlargest(top_n, "Energy")
    
    plt.figure(figsize=(18, 6))
    top_years.plot(kind="barh", **kwargs)
    plt.title(f"Top {top_n} Anos de Maior Geração de Energia Solar")
    plt.xlabel("Energia Solar (kWh)")
    plt.ylabel("Ano")
    plt.gca().invert_yaxis()
    
    for index, value in enumerate(top_years["Energy"]):
        plt.text(
            value, index, f"{value:.0f} MWh / ano",
            va="center", ha="right", fontweight="bold", fontsize=12
        )
    
    if save_image:
        export_figure(filename=f"top_{top_n}_anos.png")
    plt.show()

def plot_monthly_distribution(data: pd.DataFrame, save_image: bool = False, **kwargs) -> None:
    """
    Plots the monthly average distribution of solar energy production.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data with datetime index and energy production values.
    save_image : bool, optional
        Whether to save the generated plot as an image, by default False.
    """
    monthly_avg = data.groupby(data.index.month).mean()
    
    plt.figure(figsize=(18, 6))
    monthly_avg.plot(kind="line", **kwargs)
    plt.title("Distribuição Mensal da Energia Solar")
    plt.xlabel("Mês")
    plt.ylabel("Energia Solar (kWh)")
    plt.xticks(rotation=0)
    plt.grid(False)
    
    if save_image:
        export_figure(filename="distribuicao_mensal.png")
    plt.show()

def plot_time_series_decomposition(data: pd.DataFrame, save_image: bool = False, **kwargs) -> None:
    """
    Decomposes the time series into observed, trend, seasonal, and residual components.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data with datetime index and energy production values.
    save_image : bool, optional
        Whether to save the generated plot as an image, by default False.
    """
    decomposition = seasonal_decompose(
        x=data, model="additive", extrapolate_trend="freq", **kwargs
    )
    
    fig, axes = plt.subplots(nrows=4, figsize=(18, 12))
    fig.tight_layout()
    
    decomposition.observed.plot(ax=axes[0], ylabel="Observed")
    decomposition.trend.plot(ax=axes[1], ylabel="Trend")
    decomposition.seasonal.plot(ax=axes[2], ylabel="Seasonal")
    decomposition.resid.plot(ax=axes[3], ylabel="Residual")
    
    if save_image:
        export_figure(filename="decomposicao_temporal.png")
    plt.show()