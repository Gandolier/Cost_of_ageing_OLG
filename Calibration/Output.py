from pathlib import Path
import sys

# Make repo root importable (so `import main` works)
repo_root = Path.cwd().parent  # notebook dir -> project root
sys.path.insert(0, str(repo_root))

import pandas as pd
import numpy as np
import statsmodels.api as sm


def get_prod_params(
        path: str = '../Data/for_prod_function.xlsx',
        sheetname: str = 'real',
) -> pd.DataFrame:
    """Calibrates the parameters of the production function.

    Parameters:
        path (str): Path to the Excel file.
        sheetname (str): Sheet name

    Returns:
        A, alpha, g_y - estimated parameters of the prod function"""
    data = pd.read_excel(path, sheet_name=sheetname)
    data.set_index('Год', inplace=True)

    logdata = np.log(data)
    logdata = logdata.diff().dropna()
    y = logdata['ВВП']
    x = sm.add_constant(logdata[['ВНОК', "Занятых"]])

    GLM = sm.GLM(y, x)
    result = GLM.fit_constrained(([0, 1, 1], 1))

    alpha = result.params[1]
    g_y = result.params[0] / result.params[2]

    y_ln = np.log(data['ВВП'])
    k_ln = np.log(data['ВНОК'])
    l_ln = np.log(data['Занятых'])
    t = np.arange(data.shape[0]) + 1

    y = np.exp(
        y_ln - alpha * k_ln - (1 - alpha) * (l_ln + t * g_y)
    )

    A = np.mean(y)

    return A, alpha, g_y