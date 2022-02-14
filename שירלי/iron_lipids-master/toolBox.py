# ----------------------- imports -------------------------
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
import matplotlib.markers as markers
from sklearn.preprocessing import scale

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from numpy import mean
from numpy import absolute
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats




# -------------------- CONSTANTS --------------------
LIPLIP = "liplip"
PATH = "new_data.xlsx"
IRON = '[Fe] (mg/ml)'
LIPID = 'Lipid [fr]'
LIPID_TYPE = 'Lipid type'
IRON_TYPE = 'Iron type'
PROTEIN = "[Protein](mg/ml)"

# targets options
R1 = 'R1 [1/sec]'
R2 = 'R2 [1/sec]'
R2S = 'R2s [1/sec]'
MTV = 'MTV [fraction]'
MT = 'MT [p.u.]'

iron_type_colors = {'Fe2': 'r', 'Ferritin': 'c', 'Transferrin': 'y'}
lipid_type_markers = {'PC_SM': markers.CARETRIGHT, 'PC': '_', 'PC_Cholest': '3'}

bad_samples = [6, 8, 11, 12, 13, 14, 15, 'O1', 'O2', 'O3']

