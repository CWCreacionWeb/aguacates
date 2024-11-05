def Lista_Atributo(pDataFrame, pCampo):
    """
    Retorna una lista de valores únicos de una columna específica en un DataFrame.

    Parámetros:
    df (pd.DataFrame): El DataFrame de entrada.
    columna (str): El nombre de la columna de la que queremos los valores únicos.

    Retorna:
    list: Una lista de valores únicos en la columna especificada.
    """
    # Obtener los valores únicos y convertirlos en una lista
    vRes = pDataFrame[pCampo].unique().tolist()
    return vRes
