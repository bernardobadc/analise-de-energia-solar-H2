# Import modules
import os
import pandas as pd


def list_files(directory: os.PathLike, dataset_name: str = "Solcast") -> list:
    """
    List all files in a specified directory that contain the dataset name.

    Parameters
    ----------
    directory : os.PathLike
        Path to the directory containing the files.
    dataset_name : str, optional
        Name of the dataset to filter files (default is "Solcast").

    Returns
    -------
    list
        List of file paths that match the dataset name.
    """
    files = [os.path.join(directory, file) for file in os.listdir(directory)]
    return list(filter(lambda file_path: dataset_name in file_path, files))


def read_and_format_data(file_path: os.PathLike, **kwargs) -> pd.DataFrame:
    """
    Read and preprocess a CSV file into a formatted DataFrame.

    Parameters
    ----------
    file_path : os.PathLike
        Path to the CSV file.
    **kwargs
        Additional arguments passed to `pd.read_csv`.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with renamed columns, converted units, and adjusted values.
    """
    df = pd.read_csv(file_path, **kwargs)
    df = df.rename(
        columns={"kW": "Energy"}
    )  # Renomeia a coluna para um nome mais sugestivo
    df /= 1000  # Converte os valores de kW para MWh
    df = df.map(lambda x: 0 if x < 0 else x)  # Converte valores negativos em zero
    df.index = pd.to_datetime(
        df.index, format="%d/%m/%y %H:%M"
    )  # Converte índice para datetime
    return df


def check_file_processing(file_path: os.PathLike) -> bool:
    """
    Check if a file exists and prompt the user whether to process it again.

    Parameters
    ----------
    file_path : os.PathLike
        Path to the file.

    Returns
    -------
    bool
        True if the file should be processed, False otherwise.
    """
    if os.path.exists(file_path):
        user_response = input(
            "Arquivo já existe, deseja processar e salvar outra versão? (S/N)"
        )
        if user_response.upper() == "S":
            return True
        elif user_response.upper() == "N":
            return False
        else:
            raise ValueError("Por favor, insira uma resposta válida (S/N)")
    return True


def compile_data(
    directory: os.PathLike,
    dataset_name: str = "Solcast",
    transformer_loss: float = 0.003,
    transmission_loss: float = 0.01,
    save_data: bool = False,
) -> pd.DataFrame:
    """
    Compile and process solar energy data from multiple files.

    Parameters
    ----------
    directory : os.PathLike
        Path to the directory containing the data files.
    dataset_name : str, optional
        Name of the dataset to filter files (default is "Solcast").
    transformer_loss : float, optional
        Transformer loss factor (default is 0.003).
    transmission_loss : float, optional
        Transmission line loss factor (default is 0.01).
    save_data : bool, optional
        Whether to save the processed data to a CSV file (default is False).

    Returns
    -------
    pd.DataFrame
        Compiled and processed DataFrame containing solar energy data.
    """
    output_dir = os.path.join(os.getcwd(), "data", "output")
    output_file = os.path.join(output_dir, f"dados_compilados_{dataset_name}.csv")

    # If the file exists, ask the user whether they want to process it again
    process = check_file_processing(file_path=output_file)

    if process:
        # Search for files containing data to be concatenated
        files = list_files(directory, dataset_name)
        data_frames = []  # List to store the obtained dataframes
        for file in files:
            df = read_and_format_data(
                file,
                skiprows=11,
                sep=";",
                encoding="latin-1",
                index_col=0,
                decimal=",",
            )
            data_frames.append(df)  # Add dataframe to the list

        # Merge all tables into a single time series
        compiled_df = pd.concat(data_frames, axis=0)
        compiled_df *= (1 - transformer_loss) * (
            1 - transmission_loss
        )  # Apply energy losses

        # Save data in a CSV file
        if save_data:
            os.makedirs(output_dir, exist_ok=True)
            compiled_df.to_csv(output_file)

        return compiled_df
    else:
        return pd.read_csv(output_file, index_col=0, parse_dates=True)
