import pandas as pd


def detectar_outliers_iqr(df):
    """
    Detecta outliers en columnas numericas usando el metodo IQR.

    Retorna un DataFrame con columnas:
        variable, outliers, porcentaje, limite_inferior, limite_superior, nulos
    """
    resultados = []
    columnas_numericas = df.select_dtypes(include=['number']).columns

    for col in columnas_numericas:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        limite_inf = q1 - 1.5 * iqr
        limite_sup = q3 + 1.5 * iqr

        n_outliers = ((df[col] < limite_inf) | (df[col] > limite_sup)).sum()
        n_nulos = df[col].isnull().sum()
        porcentaje = (n_outliers / len(df) * 100).round(2)

        resultados.append({
            'variable': col,
            'outliers': n_outliers,
            'porcentaje': porcentaje,
            'limite_inferior': round(limite_inf, 2),
            'limite_superior': round(limite_sup, 2),
            'nulos': n_nulos
        })

    return pd.DataFrame(resultados)
