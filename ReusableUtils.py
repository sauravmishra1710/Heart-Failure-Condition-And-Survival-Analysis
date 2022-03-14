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
            
    def ConstructGoPieChart(self, build_sub_plots = False, num_sub_plots = 0, rows = 0, cols = 0, subplot_titles = [], 
                        labels = [], values = [], sub_plot_names = [], export_to_png = False):
    
        idx = 0
        fig = make_subplots(rows=rows, cols=cols, specs=[[{'type':'domain'}, {'type':'domain'}]],
                               subplot_titles=subplot_titles)

        for row in range(1, rows + 1):

            for col in range (1, cols + 1):

                if build_sub_plots:

                    fig.add_trace(go.Pie(labels=labels[idx], values=values[idx], 
                                         name=sub_plot_names[idx]), row=row, col=col)

                    idx += 1

        # Use `hole` to create a donut-like pie chart
        fig.update_traces(hole=.4, hoverinfo="label+percent+name")
        fig.update_layout(title_text="Hypertension Distribution & Survival Rate (SR)")

        # show the interactive view
        fig.show()
        
        self.InsertChartSeparator()
        
        # export to a png rendered format of the chart
        if export_to_png:
            fig.show("png")
    
    def Generate_Model_Test_Classification_Report(model, X_test, y_test, model_name=""):

        '''
        Purpose: 
            Generate the consolidated test classification report. 
            The report consists of the following classification results & metrics -
                1. Confusion Matrix
                2. Classification Report
                3. F1 Score
                4. Accuracy
                5. Mathews Correlation Coefficient (MCC)
                6. Precision
                7. Recall
                8. AUROC Score - Area Under the Receiver Operating Characteristic Curve
                9. AUC-PR Score - Area Under the Precision Recall Curve.
                10. AUROC Curve - Area Under the Receiver Operating Characteristic Curve
                11. AUC-PR Curve - Area Under the Precision Recall Curve.

        Parameters:
            1. y_test - The Ground Truth for each test image.
            2. y_pred - The Predicted label for each image.
            3. model_name - Model Name

        Return Value: 
            NONE.
        '''

        y = 1.05
        # Report Title & Classification Mterics Abbreviations...
        fig, axes = plt.subplots(3, 1, figsize = (8, 3))
        axes[0].text(9, 1.8, "CONSOLIDATED MODEL TEST REPORT", fontsize=30, horizontalalignment='center', 
                     color='DarkBlue', weight = 'bold')

        axes[0].axis([0, 10, 0, 10])
        axes[0].axis('off')

        axes[1].text(9, 4, "Model Name: " + model_name, style='italic', 
                             fontsize=18, horizontalalignment='center', color='DarkOrange', weight = 'bold')

        axes[1].axis([0, 10, 0, 10])
        axes[1].axis('off')

        axes[2].text(0, 4, "* 1 - Not Survived\t\t\t\t\t\t\t * 0 - Survived\n".expandtabs() +
                     "* MCC - Matthews Correlation Coefficient\t\t* AUC - Area Under The Curve\n".expandtabs() +
                     "* ROC - Receiver Operating Characteristics     " + 
                     "\t* AUROC - Area Under the Receiver Operating    Characteristics".expandtabs(), 
                     style='italic', fontsize=10, horizontalalignment='left', color='orangered')

        axes[2].axis([0, 10, 0, 10])
        axes[2].axis('off')

        scores = []
        metrics = ['F1       ', 'MCC      ', 'Precision', 'Recall   ', 'Accuracy ',
                   'AUC_ROC  ', 'AUC_PR   ']

        # Plot ROC and PR curves using all models and test data...
        y_pred = model.predict(X_test)
        y_pred_probs = model.predict_proba(X_test)[:, 1:]

        fpr, tpr, thresholds = roc_curve(y_test.values.ravel(), y_pred)
        precision, recall, th = precision_recall_curve(y_test.values.ravel(), y_pred_probs)

        # Calculate the individual classification metic scores...
        model_f1_score = f1_score(y_test, y_pred)
        model_matthews_corrcoef_score = matthews_corrcoef(y_test, y_pred)
        model_precision_score = precision_score(y_test, y_pred)
        model_recall_score = recall_score(y_test, y_pred)
        model_accuracy_score = accuracy_score(y_test, y_pred)
        model_auc_roc = auc(fpr, tpr)
        model_auc_pr = auc(recall, precision)

        scores.append([model_f1_score,
                       model_matthews_corrcoef_score,
                       model_precision_score,
                       model_recall_score,
                       model_accuracy_score,
                       model_auc_roc,
                       model_auc_pr])

        sampling_results = pd.DataFrame(columns = ['Classification Metric', 'Score Value'])
        for i in range(len(scores[0])):
            sampling_results.loc[i] = [metrics[i], scores[0][i]]

        sampling_results.index = np.arange(1, len(sampling_results) + 1)

        class_report = classification_report(y_test, y_pred)
        conf_matx = confusion_matrix(y_test, y_pred)

        # Display the Confusion Matrix...
        fig, axes = plt.subplots(1, 3, figsize = (20, 4))
        sns.heatmap(conf_matx, annot=True, annot_kws={"size": 16},fmt='g', cbar=False, cmap="GnBu", ax=axes[0])
        axes[0].set_title("1. Confusion Matrix", fontsize=21, color='darkgreen', weight = 'bold', 
                          style='italic', loc='left', y=y)

        # Classification Metrics
        axes[1].text(5, 1.8, sampling_results.to_string(float_format='{:,.4f}'.format, index=False), style='italic', 
                     fontsize=20, horizontalalignment='center')
        axes[1].axis([0, 10, 0, 10])
        axes[1].axis('off')
        axes[1].set_title("2. Classification Metrics", fontsize=20, color='darkgreen', weight = 'bold', 
                          style='italic', loc='center', y=y)

        # Classification Report
        axes[2].text(0, 1, class_report, style='italic', fontsize=20)
        axes[2].axis([0, 10, 0, 10])
        axes[2].axis('off')
        axes[2].set_title("3. Classification Report", fontsize=20, color='darkgreen', weight = 'bold', 
                          style='italic', loc='center', y=y)

        plt.tight_layout()
        plt.show()

        # AUC-ROC & Precision-Recall Curve
        fig, axes = plt.subplots(1, 2, figsize = (14, 4))

        axes[0].plot(fpr, tpr, label = f"auc_roc = {model_auc_roc:.3f}")
        axes[1].plot(recall, precision, label = f"auc_pr = {model_auc_pr:.3f}")

        axes[0].plot([0, 1], [0, 1], 'k--')
        axes[0].legend(loc = "lower right")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("4. AUC - ROC Curve", fontsize=15, color='darkgreen', ha='right', weight = 'bold', 
                          style='italic', loc='center', pad=1, y=y)

        axes[1].legend(loc = "lower left")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("5. Precision - Recall Curve", fontsize=15, color='darkgreen', ha='right', weight = 'bold', 
                          style='italic', loc='center', pad=3, y=y)

        plt.subplots_adjust(top=0.95) 
        plt.tight_layout()
        plt.show()
    
    
    def plot_model_feature_importances(model):

        '''
        Custom function to plot the 
        feature importances of the classifier.
        '''
        fig = plt.figure()

        # get the feature importance of the classifier 'model'
        feature_importances = pd.Series(model.feature_importances_,
                                index = X_train.columns) \
                        .sort_values(ascending=False)

        # plot the bar chart
        sns.barplot(x = feature_importances, y = X_train.columns)
        plt.title('Classifier Feature Importance', fontdict = {'fontsize' : 20})
        plt.xticks(rotation = 60)
        plt.show()