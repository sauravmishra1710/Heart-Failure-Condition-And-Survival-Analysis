# Import the required libraries
import numpy as np
import pandas as pd
from scipy.stats import randint 
from collections import Counter

import matplotlib.pyplot as plt 
import seaborn as sns
from statsmodels.formula.api import ols

from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as gobj
import plotly.figure_factory as ff

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.metrics import precision_score, auc, roc_auc_score, roc_curve, precision_recall_curve, classification_report
from scipy.stats import randint 

from collections import Counter



class ReusableUtils:
    
    
    """
    Module of reusable function and utilities that 
    can be reused across notebooks.
    
    """
    
    def setNotebookConfigParams():
        
        '''
        Sets the note book 
        configuration parameters.
        
        Params: None
        Return: None
        
        '''
        
        # To display all the columns
        pd.options.display.max_columns = None

        # To display all the rows
        pd.options.display.max_rows = None

        # To map Empty Strings or numpy.inf as Na Values
        pd.options.mode.use_inf_as_na = True

        pd.options.display.expand_frame_repr =  False

        # Set Style
        sns.set(style = "whitegrid")

        # Ignore Warnings
        import warnings
        warnings.filterwarnings('ignore')

        # inline plotting with the Jupyter Notebook
        init_notebook_mode(connected=True)
        
    def InsertChartSeparator():
    
        """Inserts a separator to demarcate between the dynamic interactive chart
        and the corresponding static chart in the png format."""

        print("                             ****************  STATIC PNG FORMAT  ****************") 
        
    def add_data_labels(ax, spacing = 5):

        """
        Custom Function to add 
        data labels in the graph.
        
        """
        # For each bar: Place a label
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            # Number of points between bar and label. Change to your liking.
            space = spacing
            # Vertical alignment for positive values
            va = 'bottom'

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label and format number with one decimal place
            label = "{:.2f}%".format(y_value)

            # Create annotation
            plt.annotate(
                label,                        # Use `label` as label
                (x_value, y_value),           # Place label at end of the bar
                xytext = (0, space),          # Vertically shift label by `space`
                textcoords = "offset points", # Interpret `xytext` as offset in points
                ha = 'center',                # Horizontally center label
                va = va)                      # Vertically align label differently for positive and negative values.