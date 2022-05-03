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
import plotly.figure_factory as ff

from IPython.display import display_html 

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.metrics import precision_score, auc, roc_auc_score, roc_curve, precision_recall_curve, classification_report
from scipy.stats import randint 

from collections import Counter



class ReusableUtils():
    
    
    """
    Module of reusable function and utilities that 
    can be reused across notebooks.
    
    """
    
    def __init__(self):
        pass
    
    def setNotebookConfigParams(self):
        
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
        
    def InsertChartSeparator(self):
    
        """Inserts a separator to demarcate between the dynamic interactive chart
        and the corresponding static chart in the png format."""

        print("                             ****************  STATIC PNG FORMAT  ****************")
        
        return None
        
    def add_data_labels(self, ax, spacing = 5):
        
        '''
        Purpose: 
            Custom Function to add data labels in the graph.
            
            **NOTE: A pie chart (or a circle chart) is a circular statistical graphic, 
            which is divided into slices to illustrate numerical proportion. 
            
        Parameters:
            1. build_sub_plots - Boolean flag that informs if there is a need 
            to create subplots (multiple pie charts).
            1. hist_data - Use list of lists to plot multiple data sets on the same plot.
            2. group_labels - Names for each data set.
            3. title_text - main title of the plot figure.
            4. histnorm - 'probability density' or 'probability'. Default = 'probability density'
            5. export_to_png - Boolean flag to draw a static version of the plot in png format.

        Return Value: 
            NONE.
        '''

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
            
        return None
            
    def ConstructGoPieChart(self, build_sub_plots = False, rows = 0, cols = 0, subplot_titles = [],
                            labels = [], values = [], sub_plot_names = [], title_text = "", export_to_png = False):
    
        '''
        Purpose: 
            Creates a pie charts of the specified data values.
            
            **NOTE: A pie chart (or a circle chart) is a circular statistical graphic, 
            which is divided into slices to illustrate numerical proportion. 
            
        Parameters:
            1. build_sub_plots - Boolean flag that informs if there is a need 
            to create subplots (multiple pie charts).
            2. hist_data - Use list of lists to plot multiple data sets on the same plot.
            3. group_labels - Names for each data set.
            4. title_text - main title of the plot figure.
            5. histnorm - 'probability density' or 'probability'. Default = 'probability density'
            6. export_to_png - Boolean flag to draw a static version of the plot in png format.

        Return Value: 
            NONE.
        '''
        
        if build_sub_plots:
            
            idx = 0
            fig = make_subplots(rows=rows, cols=cols, specs=[[{'type':'domain'}, {'type':'domain'}]],
                                   subplot_titles=subplot_titles)

            for row in range(1, rows + 1):

                for col in range(1, cols + 1):

                    if build_sub_plots:

                        fig.add_trace(go.Pie(labels=labels[idx], values=values[idx], 
                                             name=sub_plot_names[idx]), row=row, col=col)

                        idx += 1
                        
            # Use `hole` to create a donut-like pie chart
            fig.update_traces(hole=.3, hoverinfo="label+percent+name")
            fig.update_layout(title_text=title_text)

        else:
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0, 0.1])])
            fig.update_layout(title_text=title_text)
        
        # show the interactive view
        fig.show()
        
        # export to a png rendered format of the chart
        if export_to_png:
            self.InsertChartSeparator()            
            fig.show("png")
            
        return None
            
    def constructDistPlot(self, hist_data = [], group_labels = [], title_text = "", histnorm='probability density', 
                          colors = [], bin_size=[50, 50], export_to_png = True):
        
        '''
        Purpose: 
            Creates a distribution plot of the specified data/series.
            
            **NOTE: The distplot represents the univariate distribution of data i.e. 
            data distribution of a variable against the density distribution. 
            
        Parameters:
            1. hist_data - Use list of lists to plot multiple data sets on the same plot.
            2. group_labels - Names for each data set.
            3. title_text - main title of the plot figure.
            4. histnorm - 'probability density' or 'probability'. Default = 'probability density'
            5. colors - Colors for traces.
            6. bin_size - Size of histogram bins.
            7. export_to_png - Boolean flag to draw a static version of the plot in png format.

        Return Value: 
            NONE.
        '''
    
        fig = ff.create_distplot(hist_data=hist_data,
                                 group_labels=group_labels,
                                 bin_size=bin_size, 
                                 colors = colors, 
                                 histnorm=histnorm)
        
        fig.update_layout(title_text = title_text)

        # show the interactive view
        fig.show()

        # export to a png rendered format of the chart
        if export_to_png:
            self.InsertChartSeparator()
            fig.show("png")
            
        return None
    
    def constructPxHistogram(self, data_frame, x, color, marginal, 
                             hover_data, title, export_to_png = False):
    
        '''
        Purpose: 
            Creates a histogram distribution plot of the specified data/series.

            **NOTE: A histogram is representation of the distribution of numerical data, 
            where the data are binned and the count for each bin is represented. 
            In a histogram, rows of `data_frame` are grouped together into a
            rectangular mark to visualize the 1D distribution of an aggregate
            function `histfunc` (e.g. the count or sum) of the value `y` (or `x` if
            `orientation` is `'h'`). 
            Ref - DocString for px.histogram

        Parameters:
            1. data_frame - DataFrame or array-like or dict data required for the histogram.
            2. x - Names for each data set. str or int or Series or array-like
                   Either a name of a column in `data_frame`, or a pandas Series or
                   array_like object. 
            3. color - str or int or Series or array-like
                       Either a name of a column in `data_frame`, or a pandas Series or
                       array_like object. Values from this column or array_like are used to
                       assign color to marks.
            4. marginal - (str) One of `'rug'`, `'box'`, `'violin'`, or `'histogram'`. If set, a
                           subplot is drawn alongside the main plot, visualizing the distribution.
            5. title - Colors for traces.
            6. export_to_png - Boolean flag to draw a static version of the plot in png format.

        Return Value: 
            NONE.
        '''

        fig = px.histogram(data_frame = data_frame,
                           x = x, 
                           color = color,
                           marginal = marginal, 
                           hover_data = data_frame.columns,
                           title = title)

        # show the interactive view
        fig.show()

        # export to a png rendered format of the chart
        if export_to_png:
            self.InsertChartSeparator()
            fig.show("png")
            
        return None
    
    def constructNotchedBoxPlots(self, data_frame, x, y, hover_name, color, title_text,
                                 points = 'all', export_to_png = False):
        
        '''
        Purpose: 
            Creates a notched box plot distribution of the specified data/series.

            **NOTE: Notched box plots apply a "notch" or narrowing of the box around the median. 
              Notches are useful in offering a rough guide to significance of difference of medians; 
              if the notches of two boxes do not overlap, this offers evidence of a statistically significant 
              difference between the medians.
              Ref - https://en.wikipedia.org/wiki/Box_plot#Variations


        Parameters:
            1. data_frame - DataFrame or array-like or dict data required for the histogram.
            2. x - (str or int or Series or array-like) Either a name of a column in `data_frame`, 
                   or a pandas Series or array_like object. 
            3. y - (str or int or Series or array-like) – Either a name of a column in data_frame, 
                    or a pandas Series or array_like object. Values from this column or array_like are used to 
                    position marks along the y axis in cartesian coordinates. 
            4. hover_name - (str or int or Series or array-like) – Either a name of a column in data_frame, 
                            or a pandas Series or array_like object. Values from this column or array_like 
                            appear in bold in the hover tooltip.
            5. color - Either a name of a column in data_frame, or a pandas Series or array_like object. 
                       Values from this column or array_like are used to assign color to marks.
            6. title_text - Title of the plot figure.
            7. export_to_png - Boolean flag to draw a static version of the plot in png format.
            
            Ref: https://plotly.github.io/plotly.py-docs/generated/plotly.express.box.html

        Return Value: 
            NONE.
        '''
        
        fig = px.box(data_frame, 
                     x = x, 
                     y = y, 
                     points = points, 
                     hover_name = hover_name, 
                     color = color, 
                     notched=True)

        fig.update_layout(title_text = title_text)

        # show the interactive view
        fig.show()

        # export to a png rendered format of the chart
        if export_to_png:
            self.InsertChartSeparator()
            fig.show("png")
            
        return None
    
    def plotUnivariateAnalysis(self, data_frame, category_list, rows, cols, figsize = (8, 8)):
        
        '''
        Purpose: 
            Plots the univariate analysis of the given categorical variables.

        Parameters:
            1. data_frame = the master dataframe.
            2. category_list - the list of categorical variables
            3. rows - Number of rows in the subplots.
            4. cols - Number of columns in the subplots.
            5. figsize - Size of the plot figure.

        Return Value: 
            NONE.
        '''
    
        counter = 1

        plt.figure(figsize = figsize)
        
        for col_list in category_list:

            series = round(((data_frame[col_list].value_counts(dropna = False))/
                            (len(data_frame[col_list])) * 100), 2)

            plt.subplot(rows, cols, counter)
            ax = sns.barplot(x = series.index, y = series.values, order = series.sort_index().index)
            sns.despine(bottom = True, left = True)
            plt.xlabel(col_list, labelpad = 15)
            plt.ylabel('Percentage Rate', labelpad = 10)

            ax.grid(False)
            
            # Call Custom Function
            self.add_data_labels(ax)

            counter += 1

        del category_list, counter, ax
        
        plt.subplots_adjust(hspace = 0.3)
        plt.subplots_adjust(wspace = 0.5)
        plt.show()

        return None
    
    def plotDataCorrelationHeatMap(self, data_frame, fig_size = (15,10)): 
    
        '''
        Purpose:
            Plots the data / feature correlation heatmap.

        Parameters:
            1. data_frame = the master dataframe.
            2. figsize - Size of the plot figure. Default Size is set to (15, 10)

        Return Value: 
            corr -  The data correlation matrix.

        '''

        fig, ax = plt.subplots(figsize = fig_size)
        
        corr = data_frame.corr()
        sns.heatmap(corr, annot = True, linewidths = .5, ax = ax)
        plt.show()

        return corr
    
    def Generate_Model_Test_Classification_Report(self, model, X_test, y_test, model_name=""):

        '''
        Purpose: 
            Generate the consolidated test classification report. 
            A one-stop function to generate all the relevant model evaluation metrics. 
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
        
        return None
    
    def plot_model_feature_importances(self, X_train, model):

        '''
        Purpose: 
            Custom function to plot the feature importances of the classifier.
            
            **NOTE: Feature importances specify how much each feature is contributing 
            towards the final prediction value/results. 
            
        Parameters:
            1. model - the model whose feature importances are to be plotted.
            2. X_train - Training dataset.

        Return Value: 
            NONE.
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
        
        return None
    
    def display_dataframe_side_by_side(self, dataframes:list, table_captions:list, master_caption = None, tablespacing=5):
    
        """
        Purpose:
            Display the dataframes as html tables side by side 
            to save vertical space.

        Parameters:
            dataframes: list of pandas.DataFrame
            table_captions: list of table captions
            tablespacing: table separator to differentiate the tables/dataframes. 
                          The vertical spacing between tables..

        Returns:
            NONE

        Reference:
            https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
        """

        output = ""
        space_char = "\xa0"
        
        if master_caption is not None:
            master_caption = '<h2 style="color: black;">' + master_caption + '</h2>'
        
        display_html(master_caption, raw = True)
        
        # table caption styler
        styler = [dict(selector="caption",
                       props=[("text-align", "left"),
                              ("font-size", "125%"),
                              ("text-decoration", "underline"),
                              ("color", 'red')]), 
          dict(selector="td", props=[('font-size', '100%')])]

        for (df, caption) in zip(dataframes, table_captions):
            output += df.style.set_table_attributes("style='display:inline;'")\
            .set_caption(caption)\
            .set_table_styles(styler)\
            ._repr_html_()
            
            output += tablespacing * space_char
            
        display_html(output, raw = True)

        return None