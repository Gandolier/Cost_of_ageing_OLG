from pathlib import Path
import sys

# Make repo root importable (so `import main` works)
repo_root = Path.cwd().parent  # notebook dir -> project root
sys.path.insert(0, str(repo_root))

from main import reimport
import warnings
warnings.filterwarnings('ignore')

from Steady_state_equilibrium.SS_Solver import SteadyStateEquilibrium

import numpy as np
import pandas as pd
import scipy.interpolate as si
from matplotlib import pyplot as plt


def get_omega(
        path: str = '../Data/total_pop.xlsx',
        unit: int = 1000,
        normalize: bool = True,
        E: int = 20,
) -> pd.DataFrame:
    """Retrieve and preprocess population data from an Excel file.

    Parameters:
        path (str): Path to the Excel file containing population data.
        unit (int): Conversion factor for population units.
        normalize (bool): Whether to normalize population data to a unit.

    Returns:
        pd.DataFrame: Preprocessed population data with 'Year' as index
        and population in millions or normalized to share of total population (sums up to 1).
    """
    omega = pd.read_excel(path)

    omega.set_index('Year', inplace=True)
    omega /= (1_000_000 // unit) # now in mils

    if normalize:
        pop_base = omega.iloc[:, E:].sum(axis=1)
        omega = omega.div(pop_base, axis=0)

    return omega

def get_g_n(
        omega: pd.DataFrame,
        end_year: int = 2025,
) -> float:
    """Calculate the growth rate of population from one year to another.

    Parameters:
        omega (pd.DataFrame): Population data with 'Year' as index.
        end_year (int): The year to calculate the growth rate for.

    Returns:
        float: The growth rate of population from the previous year to the specified year.
    """
    return float(omega.loc[end_year].sum() / omega.loc[end_year-1].sum() - 1)

def get_mortality(
        path: str = '../Data/death_probability.xlsx',
        totpers: int = 100,
) -> pd.DataFrame:
    """Retrieve and preprocess mortality data from an Excel file.

    Parameters:
        path (str): Path to the Excel file containing mortality data.
        totpers (int): Total number of persons in the population.

    Returns:
        pd.DataFrame: Preprocessed mortality data with 'Year' as index.
    """
    data = pd.read_excel(path)
    data.set_index('Year', inplace=True)
    mort = pd.DataFrame(index=data.index, columns=np.arange(totpers) + 1)

    for i, year in enumerate(data.index):
        mort_data = data.iloc[i, :]
        mort_data.loc[-1] = 0.
        mort_data.loc[-2] = 0.
        mort_data.loc[totpers+1] = 1.
        mort_data.loc[totpers+2] = 1.

        age_midp = mort_data.index.values

        fit_mort = si.interp1d(age_midp, mort_data, kind='cubic', bounds_error=False, fill_value=0)

        age_model = np.linspace(100 / totpers, 100, totpers) - (0.5 * 100 / totpers)
        mort_model = fit_mort(age_model)
        mort.loc[year] = mort_model

    return mort[(mort >= 0.) & (mort <= 1.)].ffill(axis=1)

def get_tot_migration(
        path: str = '../Data/age specific migration.xlsx',
        sheetname: str = 'migration',
        unit: int = 1000,
        normalize: bool = True,
        E: int = 20,
        omega: pd.DataFrame = None,
) -> pd.DataFrame:
    """Retrieve and preprocess migration data from an Excel file."""
    I = pd.read_excel(path, sheet_name=sheetname).set_index('Year')
    I /= (1_000_000 // unit) # now in mils

    if normalize:
        pop_base = omega.iloc[:, E:].sum(axis=1)
        I = I.div(pop_base, axis=0)

    return I

def get_migration_rate(
        path: str = '../Data/age specific migration.xlsx',
        sheetname: str = 'migration',
) -> pd.Series:
    """Calculate migration rates from total migration data."""
    i = pd.read_excel(path, sheet_name='rate').set_index('age')
    return i['average']