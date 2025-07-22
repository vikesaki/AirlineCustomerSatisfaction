import pandas as pd
import numpy as np
from scipy.stats import kurtosis, chi2_contingency, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import silhouette_score, silhouette_samples
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from sklearn.model_selection import cross_val_score

def analyze_distribution(dataset, target = None):
    """
    Analyze skewness of numerical columns in dataset.

    Parameters:
        dataset (pd.DataFrame): Training data containing numerical features.

    Returns:
        pd.DataFrame: Skewness values, Kurtosis value and distribution type.
    """
    if target:
        try:
            nums = target
        except:
            print('"multi" set as True, means variable should be more than one')
            return
    else :
        nums = list(dataset.select_dtypes(exclude='object').columns)

    data = []
    for col in nums:
        skew_val = round(dataset[col].skew(), 1)
        if skew_val > 1 or skew_val < -1:
            skew_label = 'highly skewed'
        elif 0.5 < abs(skew_val) <= 1:
            skew_label = 'skewed'
        else:
            skew_label = 'normal'
        
        kurt_val = round(kurtosis(dataset[col]), 2) 
        if -2 <= kurt_val <= 2:
            kurt_label = 'normal'
        elif kurt_val > 2:
            kurt_label = 'heavy tails'
        else:
            kurt_label = 'light tails'
            
        data.append([col, skew_val, skew_label, kurt_val, kurt_label])

    distribution_df = pd.DataFrame(
        data,
        columns=['column name', 'skewness value', 'distribution', 'kurt_val', 'kurt_label']
    )

    return distribution_df


def analyze_outliers(dataset, distribution_df):
    """
    Analyze outliers based on skewness distribution per column.

    Parameters:
        dataset (pd.DataFrame): Training data.
        distribution_df (pd.DataFrame): Skewness results from analyze_skewness().

    Returns:
        pd.DataFrame: Outlier boundaries and percentage per column.
    """
    column = []
    lower_bound = []
    upper_bound = []
    percent_total_outlier = []

    for _, row in distribution_df.iterrows():
        col = row['column name']

        if row['distribution'] in ['skewed', 'highly skewed']:
            IQR = dataset[col].quantile(0.75) - dataset[col].quantile(0.25)
            lb = dataset[col].quantile(0.25) - 3 * IQR
            ub = dataset[col].quantile(0.75) + 3 * IQR
        else:
            lb = dataset[col].mean() - 3 * dataset[col].std()
            ub = dataset[col].mean() + 3 * dataset[col].std()

        total_outliers = ((dataset[col] > ub).sum() + (dataset[col] < lb).sum())
        percent_outliers = (total_outliers / len(dataset)) * 100

        column.append(col)
        lower_bound.append(round(lb, 2))
        upper_bound.append(round(ub, 2))
        percent_total_outlier.append(round(percent_outliers, 2))

    outliers_df = pd.DataFrame({
        'column': column,
        'lower_boundary': lower_bound,
        'upper_boundary': upper_bound,
        'percentage_total_outlier': percent_total_outlier
    })

    return outliers_df


def diagnostic_plots(df, variable):
    # Define figure size
    plt.figure(figsize=(16, 4))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df[variable], bins=30)
    plt.title(f'Distribution of {variable}')

    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[variable])
    plt.title(f'Boxplot of {variable}')

    plt.show()
    
def distribution_plots(df, variable):
    plt.figure(figsize=(10, 3))

    if df[variable].dtype == 'object' or df[variable].nunique() < 15:
        sns.countplot(data=df, x=variable, order=df[variable].value_counts().index)
        plt.title(f'Count Plot of {variable}')
        plt.xticks(rotation=45)
    else:
        sns.histplot(df[variable], kde=True, bins=30)
        plt.title(f'Distribution of {variable}')

    plt.tight_layout()
    plt.show()
    
def cardinality(df, variable, multi = False):
    listItem = []
    if multi :
        try:
            for col in variable:
                listItem.append([col, df[col].nunique(), df[col].unique()])
        except:
            print('"multi" set as True, means variable should be more than one')
            return
    else :
        try:
            listItem.append([variable, df[variable].nunique(), df[variable].unique()])
        except:
            print('"multi" set as False, means variable should be only one of the column')
            return

    return pd.DataFrame(listItem, columns=['column name', 'number of unique', 'unique value'])

def correlation_matrix (df, variables=None, method='pearson', cmap="coolwarm", target=None, annotation = "horizontal"):
    """
    Generates and visualizes a correlation matrix heatmap from the given DataFrame.

    If a target column is specified, the function will display only the correlation
    between the target and the other selected variables.

    Args:
        df (pd.DataFrame): The input dataset containing numeric features.
        variables (list, optional): List of column names to include in the correlation matrix.
            If None, all numeric columns in the DataFrame are used.
        method (str, optional): Correlation method to use.
            One of {'pearson', 'spearman', 'kendall'}. Default is 'pearson'.
        cmap (str, optional): Colormap used in the heatmap. Default is 'coolwarm'.
        target (str, optional): If provided, display only correlations with this column.

    Returns:
        None: Displays a seaborn heatmap of the correlation matrix.

    Raises:
        Prints error message if an exception occurs.
    """
    try :
        if variables is None:
            variables = list(df.select_dtypes(include='number'))
        matrix = df[variables].corr(method=method)

        if target :
            matrix = matrix[[target]].T
            
        if annotation == 'horizontal':
            rotation = 0
        elif annotation == 'vertical':
            rotation = 90
        else :
            print(f'Incorrect input: {annotation}, reverting to default')
            rotation = 0
            return

        #Visualize correlation matrix as heatmap,
        plt.figure(figsize=(10, 5))
        sns.heatmap(matrix, annot=True, cmap=cmap, fmt=".3f", annot_kws={"rotation": rotation})
        plt.title("Correlation Matrix")
        plt.show()
        
    except Exception as e:
        print(f'Error occurred: {e}')
        return
    
def KDE_comparison(original, filled, column):
    """
    Generate a KDE graph of a column before and after imputation.

    Args:
        original (pd.DataFrame): Original DataFrame with NaN values.
        filled (pd.DataFrame): DataFrame after imputation.
        column (str): Column to analyze.

    Returns:
        None: Displays a seaborn heatmap of the correlation matrix.
    
    Raises:
        Prints error message if an exception occurs.
    """
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        original[column].plot(kind='kde', label='Original (with NaN value)', ax=ax)
        filled[column].plot(kind='kde', label='KNN Imputed', ax=ax)

        ax.set_title(f"KDE Comparison of {column} column Before and After")
        ax.legend(loc='best')
        plt.xlabel("Bedrooms")
        plt.xlim(min(original[column].min(), filled[column].min()) - 2, max(original[column].max(), filled[column].max()) + 2)
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f'Error occurred: {e}')
        return
    
def statistic_comparison(original, filled, column):
    """
    Compare descriptive statistics, skewness, and kurtosis of a column before and after imputation.

    Args:
        original (pd.DataFrame): Original DataFrame with NaN values.
        filled (pd.DataFrame): DataFrame after imputation.
        column (str): Column to analyze.

    Returns:
        pd.DataFrame: Summary table comparing original and imputed statistics.
    
    Raises:
        Prints error message if an exception occurs.
    """
    try:
        stats_original = original[column].describe()
        stats_imputed = filled[column].describe()

        summary = pd.DataFrame({
            'original': stats_original,
            'imputed': stats_imputed
        })

        # Add skewness and kurtosis
        summary.loc['skew'] = [original[column].skew(), filled[column].skew()]
        summary.loc['kurtosis'] = [original[column].kurtosis(), filled[column].kurtosis()]

        return summary
    
    except Exception as e:
        print(f'Error occurred: {e}')
        return
    
# NAN HANDLING SECTION

def missing_data_percentage(df):
    """
    Returns a DataFrame showing the count and percentage of missing values per column.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        pd.DataFrame: A summary of missing values.
    """
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df)) * 100

    summary = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing %': missing_percent
    })
    summary = summary[summary['Missing Count'] > 0]
    return summary


def missing_feature_analyze(df, target = None):
    """
    DOESNT WORK DONT USE
    For each column with missing values, compare the mean of all numeric columns
    with dataset with NaN and without NaN
    Also return the amount of NaN and their percentage

    Parameters
    ----------
    df (pd.DataFrame): Input DataFrame.
    target (str, optional): If provided, display only this column.

    Returns
    -------
    pd.DataFrame: with the calculated data.
    """
    try:
        if target:
            nanumns = [target]
        else:
            nanumns = df.columns[df.isna().any()].tolist()

        result = []

        for col in nanumns:
            total_rows = len(df)
            missing_count = df[col].isna().sum()
            missing_percentage = 100 * missing_count / total_rows

            df_nan = df[df[col].isna()]
            df_not_missing = df[df[col].notna()]

            means_missing = df_nan.mean(numeric_only=True)
            means_not_missing = df_not_missing.mean(numeric_only=True)

            diff = (means_not_missing - means_missing) / means_not_missing * 100
            diff = diff[diff.notna() & (diff != 0)]

            row = {
                'missing_count': missing_count,
                'missing_percentage': round(missing_percentage, 2),
            }

            row.update(diff.to_dict())
            result.append(pd.Series(row, name=col))

        return pd.DataFrame(result)

    except Exception as e:
        print(f'Error occurred: {e}')
        return
    
def missing_feature_analyze_means(df: pd.DataFrame, target = None):
    """
    For each column with missing values (or a specified target), compares the mean of all numeric features
    between rows with and without missing values. Returns only the difference where there is a change.
    Also includes missing count and missing percentage.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target : str, optional
        Column to analyze. If not provided, all columns with missing values are analyzed.

    Returns
    -------
    pd.DataFrame
        DataFrame where:
        - Each row reno_nans a column with missing values.
        - Columns include only features with a difference in mean,
          along with 'missing_count' and 'missing_percentage'.
    """
    try:
        if target:
            nanumns = [target]
        else:
            nanumns = df.columns[df.isna().any()].tolist()

        result = []

        for col in nanumns:
            total_rows = len(df)
            missing_count = df[col].isna().sum()
            missing_percentage = 100 * missing_count / total_rows

            df_nan = df[df[col].isna()]
            df_not_missing = df[df[col].notna()]

            means_missing = df_nan.mean(numeric_only=True)
            means_not_missing = df_not_missing.mean(numeric_only=True)

            diff = means_not_missing + means_missing
            diff = diff[diff.notna() & (diff != 0)]

            row = {
                'missing_count': missing_count,
                'missing_percentage': round(missing_percentage, 2),
            }

            row.update(diff.to_dict())
            result.append(pd.Series(row, name=col))

        return pd.DataFrame(result)

    except Exception as e:
        print(f'Error occurred: {e}')
        return
    
def column_summary_by_missing(df, target, stat='mean'):
    """
    Compute summary statistic for numerical columns based on NaN in a target categorical column.
    stat: 'mean', 'median', 'count', 'nunique'
    """
    # df_nan = df[df[target].isna()]
    df_no_nann = df[df[target].notna()]
    df_nan = df
    
    # print (df_nan)
    # print (df_no_nann)
    # print (df_all)
    
    nums = df.select_dtypes(include='number').columns
    result = []

    for col in nums:
        if stat == 'mean':
            val_nan = df_nan[col].mean()
            val_no_nann = df_no_nann[col].mean()
        elif stat == 'median':
            val_nan = df_nan[col].median()
            val_no_nann = df_no_nann[col].median()
        elif stat == 'count':
            val_nan = df_nan[col].count()
            val_no_nann = df_no_nann[col].count()
        elif stat == 'nunique':
            val_nan = df_nan[col].nunique()
            val_no_nann = df_no_nann[col].nunique()
        else:
            raise ValueError("Unsupported stat type.")
        
        diff = val_nan - val_no_nann
        perc_diff = ((diff / val_no_nann) * 100) if val_no_nann != 0 else np.nan
        
        result.append({
            'feature': col,
            f'{stat}_with_nan': val_nan,
            f'{stat}_without_nan': val_no_nann,
            'difference': diff,
            'percentage_difference': perc_diff
        })

    return pd.DataFrame(result).set_index('feature')

def categorical_summary_by_missing(df, target):
    """
    Compute number of unique values and mode for each categorical column
    based on whether the target column is missing.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target : str
        Target column to check for missingness

    Returns:
    --------
    pd.DataFrame
        Summary of nunique and mode when target is missing vs not missing
    """
    df_nan = df
    df_no_nan = df[df[target].notna()]
    
    cat_cols = df.select_dtypes(include='object').columns
    result = []

    for col in cat_cols:
        nunique_missing = df_nan[col].nunique()
        nunique_no_nan = df_no_nan[col].nunique()

        try:
            mode_missing = df_nan[col].mode().iloc[0]
        except IndexError:
            mode_missing = None

        try:
            mode_no_nan = df_no_nan[col].mode().iloc[0]
        except IndexError:
            mode_no_nan = None

        result.append({
            'feature': col,
            'nunique_with_nan': nunique_missing,
            'nunique_with_no_nan': nunique_no_nan,
            'mode_with_nan': mode_missing,
            'mode_with_no_nan': mode_no_nan
        })

    return pd.DataFrame(result).set_index('feature')

def cramers_v(x, y):
    """
    Computes Cramér's V statistic for categorical-categorical association.

    Args:
        x (pd.Series): First categorical variable.
        y (pd.Series): Second categorical variable.

    Returns:
        float: Cramér's V value (0 to 1).
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    if min_dim == 0:
        return np.nan  # Avoid division by zero for 1xN or Nx1 contingency tables
    return np.sqrt((chi2 / n) / min_dim)

def categorical_association_test(df, cat_columns, target_col):
    """
    Performs Chi-Square test of independence between each categorical column and the target column.
    
    Parameters:
    - df: DataFrame
    - cat_columns: list of categorical column names to test
    - target_col: name of the categorical column to test association with
    
    Returns:
    - DataFrame with columns: column name, p-value, and interpretation
    """
    results = []

    for col in cat_columns:
        contingency_table = pd.crosstab(df[col], df[target_col])
        
        chi2, p_value, dof, ex = chi2_contingency(contingency_table)
        
        interpretation = 'associated' if p_value < 0.05 else 'not associated' # type: ignore
        v = cramers_v(df[col], df[target_col])
        
        results.append({
            'column': col,
            'chi2 score': round(chi2, 2), # type: ignore
            'p_value': p_value,
            'cramers_v': v,
            'interpretation': interpretation
        })

    return pd.DataFrame(results)


# MODEL REVIEW SECTION
    
def scatter_model_prediction(prediction, target, log=False) :
    try :
        if not log :
            plt.figure(figsize=(6, 6))
            plt.scatter(target, prediction, alpha=0.5)
            plt.plot([target.min(), target.max()], [target.min(), target.max()], 'r--')
            plt.xlabel(f"Actual y")
            plt.ylabel(f"Predicted y")
            plt.title("Predicted vs Actual (log scale)")
            plt.grid(True)
            plt.show()
        else :
            plt.figure(figsize=(6, 6))
            plt.scatter(np.expm1(target), np.expm1(prediction), alpha=0.5)
            plt.plot([np.expm1(target).min(), np.expm1(target).max()],
                    [np.expm1(target).min(), np.expm1(target).max()], 'r--')
            plt.xlabel(f"Actual log(y)")
            plt.ylabel(f"Predicted log(y)")
            plt.title("Predicted vs Actual (original scale)")
            plt.grid(True)
            plt.show()
                  
    except Exception as e:
            print(f'Error occurred: {e}')
            return
        
def residual_model_prediction(prediction, target, log=False) :
    try :
        if not log :
            residuals = target - prediction
        else :
            residuals = np.expm1(target) - np.expm1(prediction)
            
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=prediction, y=residuals, alpha=0.7)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.title('Residual Plot')
        plt.grid(True)
        plt.show()
                  
    except Exception as e:
            print(f'Error occurred: {e}')
            return
        
def evaluate_model(target, prediction, log=False):
    """
    Evaluate model predictions and return a DataFrame of metrics.

    Parameters:
    - target: true target values (array-like)
    - prediction: predicted target values (array-like)
    - log: bool, if True assumes target and prediction are in log-space and evaluates both in log and original space
    - dataset_label: str, label to identify dataset (e.g. 'train', 'test')

    Returns:
    - pandas DataFrame with evaluation metrics
    """
    
    results = []
    try :
        if log:
            # Log-space metrics
            mae_log = mean_absolute_error(target, prediction)
            rmse_log = root_mean_squared_error(target, prediction)
            r2_log = r2_score(target, prediction)

            # Convert back to original scale
            target_exp = np.expm1(target)
            prediction_exp = np.expm1(prediction)
        else:
            mae_log = rmse_log = r2_log = None
            target_exp = target
            prediction_exp = prediction

        # Original-space metrics
        mae = mean_absolute_error(target_exp, prediction_exp)
        rmse = root_mean_squared_error(target_exp, prediction_exp)
        mape = np.mean(np.abs((target_exp - prediction_exp) / target_exp)) * 100
        r2 = r2_score(target_exp, prediction_exp)

        # Compose output with metrics side-by-side
        results = {
            "MAE": mae,
            "MAE in log": mae_log,
            # "RMSE": rmse,
            # "RMSE in log": rmse_log,
            "R2": r2,
            "R2 in log": r2_log,
            "MAPE": mape
        }

        df = pd.DataFrame(results, index=["metrics"])
        return df
    
    except Exception as e:
            print(f'Error occurred: {e}')
            return
        
# Assumption Test
def multicol_vif(X, target):
    try:
        X = X.drop([target], axis=1)
    except:
        print(f'Target doesnt exist!')
        
    cols = X.select_dtypes(include='number').columns
    X = X.select_dtypes(include='number')
    
    vif = pd.DataFrame()
    vif["variables"] = cols
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

def linear_relationship(df, target=None):
    
    try:
        x = df.drop([target], axis=1)
    except:
        x = df
        print(f'Target doesnt exist!')
    
    cols = x.select_dtypes(include='number').columns
    x = x.loc[:, ~x.columns.isin(cols)]
    
    fig = plt.figure(figsize=(20, 20))
    
    num_cols = 3
    num_rows = int(len(cols) / num_cols)
    
    for index in range(1, num_rows*num_cols + 1):
        fig.add_subplot(num_rows, num_cols, index)
        sns.scatterplot(data=df, x=cols[index-1], y=target)
        
    plt.show()
    
    
def coefficient_interpretation(features, coefficients, intercept):
    """
    features: list of feature names (str)
    coefficients: list or array of float log-coefficients from model
    intercept: value of intercept
    
    in general the rule is
    intercept + coefficient * (feature amount)
    
    in case anyone read this. this. is. annoying.
    """
    result_lines = []
    price_minimum = np.expm1(intercept)
    result_lines.append(
            f"at minimum, the house price will be : {price_minimum:,.2f}"
        )
    for feat, coef in zip(features, coefficients):
        price_change =  coef
        result_lines.append(
            f"for {feat} feature, for every feature increase, price increase or decrease by : {price_change:,.2f} * amount of feature"
        )
        
    return "\n".join(result_lines)

def duplicate_feature_analysis(df, target_col):
    feature_cols = list(df.columns)
    feature_cols.remove(target_col)
    unique_target = df[target_col].unique()
    
    grouped = df.groupby(feature_cols)[target_col].value_counts().unstack(fill_value=0)
    grouped = grouped.rename(columns={'will_buy_on_return_visit' : 'count'})
    
    grouped['total count'] = grouped.sum(axis=1)
    grouped['one label'] = grouped[[val for val in unique_target]].max(axis=1) == grouped['total count']
    
    for name in unique_target:
        newname = str(name) + ' count'
        grouped = grouped.rename(columns={name: newname})
        
    grouped = grouped[grouped['total count'] > 1]
    # Reset index to turn back into a DataFrame
    return grouped.reset_index()


# TLDR, im suck at coding why in the world my code doesnt work and chat gpt does, gg
'''def drop_redundant_one_label_duplicates(original_df, duplicated_df, target_col):
    label_name = list(duplicated_df.columns)
    non_label = list(original_df.columns)
    
    for name in non_label:
        if name in label_name:
            label_name.remove(name)
    
    # print(f'unique label - {label_name}')
    
    redundant_keys = duplicated_df[(duplicated_df['one label'] == True)]
    
    # Get the feature columns used for grouping
    feature_cols = [col for col in duplicated_df.columns if col not in label_name]
    
    # Merge to mark rows to drop
    merge_cols = feature_cols
    redundant_rows = original_df.merge(redundant_keys[merge_cols], on=merge_cols, how='left')
    
    # Remove
    df_cleaned = original_df.drop(redundant_rows.index)

    deduped = redundant_rows.drop_duplicates(subset=merge_cols)

    # Final cleaned data
    df_final = pd.concat([df_cleaned, deduped])

    return df_final'''

# i dont see any different between my code and chatgpt ones, but this one SOMEHOW work
def drop_redundant_one_label_duplicates(original_df, duplicated_df, target_col):
    # WOW MY OWN CODE SECTION IM SOOOOO PROUD
    label_name = list(duplicated_df.columns)
    non_label = list(original_df.columns)
    
    for name in non_label:
        if name in label_name:
            label_name.remove(name)
    feature_cols = [col for col in duplicated_df.columns if col not in label_name]

    # Filter groups with one label True and total count > 1 (actual duplicates)
    redundant_keys = duplicated_df[(duplicated_df['one label'] == True) & (duplicated_df['total count'] > 1)]

    # Reset index in original_df to preserve indices after merge
    original_df_reset = original_df.reset_index()

    # Merge on feature columns with inner join to get redundant rows
    redundant_rows = original_df_reset.merge(redundant_keys[feature_cols], on=feature_cols, how='inner')

    # Indices of rows to drop from original dataframe
    rows_to_drop = redundant_rows['index']

    # Drop all redundant rows from original
    df_cleaned = original_df_reset.drop(rows_to_drop)

    # Add back exactly one row per duplicate group (deduplicate by feature columns)
    deduped = redundant_rows.drop_duplicates(subset=feature_cols)

    # Combine cleaned data and one representative row per group
    df_final = pd.concat([df_cleaned, deduped], ignore_index=True)

    # If you want to restore original index:
    df_final = df_final.drop(columns=['index'])

    return df_final

def roc_auc(model, x, y) :
    y_scores = model.predict_proba(x)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_scores)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, y_scores):.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
    
    
def roc_auc_comparison(model, x_train, y_train, x_test, y_test, positive_class = 0) :
    y_scores_train = model.predict_proba(x_train)[:, positive_class]
    y_scores_test = model.predict_proba(x_test)[:, positive_class]
    
    fpr_train, tpr_train, thresholds = roc_curve(y_train, y_scores_train)
    auc_train = roc_auc_score(y_train, y_scores_train)
    
    fpr_test, tpr_test, thresholds = roc_curve(y_test, y_scores_test)
    auc_test = roc_auc_score(y_test, y_scores_test)
    
    intepretation = []
    for score in [auc_train, auc_test] :
        if score > 0.9 :
            intepretation.append("Good")
        elif 0.9 >= score >= 0.8 :
            intepretation.append("Fair")
        elif 0.8 > score >= 0.7 :
            intepretation.append("Decent")
        elif 0.7 > score >= 0.6 :
            intepretation.append("Not Great")
        else: 
            intepretation.append("Poor")
    
    results = pd.DataFrame({
    "Dataset": ['Train', 'Test'],    
    "AUC": [f"{auc_train:.2f}", f"{auc_test:.2f}"],
    "Intepretation": intepretation
    })

    print(results)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr_train, tpr_train, label=f"Train AUC = {auc_train:.2f}")
    plt.plot(fpr_test, tpr_test, label=f"Test AUC = {auc_test:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
    
    
"""def probability_sigmoid(model, x, y, target):
    all_features = x.select_dtypes(include='number').columns
    base_input = x.select_dtypes(include='number').mean().to_frame().T

    for feature in all_features:
        x_vals = np.linspace(x[feature].min(), x[feature].max(), 300)
        input_data = pd.concat([base_input] * len(x_vals), ignore_index=True)
        input_data[feature] = x_vals

        y_probs = model.predict_proba(input_data)[:, 1]

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_probs, color='red', label='Logistic regression curve')
        plt.scatter(x[feature], y, alpha=0.2, label='Training data (projected)')
        plt.xlabel(feature)
        plt.ylabel('Probability of Will Buy (Class 1)')
        plt.title(f'Effect of {feature} on {target} Probability')
        plt.legend()
        plt.grid(True)
        plt.show()"""
        
def probability_sigmoid(model, x, y, target):

    # Loop through each numeric column one at a time
    numeric_columns = x.select_dtypes(include='number').columns

    for feature in numeric_columns:
        # Start from one full example with all columns present
        base_input = x.iloc[[0]].copy()

        # Create a range of values for the feature being analyzed
        x_vals = np.linspace(x[feature].min(), x[feature].max(), 300)

        # Duplicate the row to match number of test points
        input_data = pd.concat([base_input] * len(x_vals), ignore_index=True)
        input_data[feature] = x_vals  # Modify only the selected feature

        # Predict probabilities using the full pipeline
        y_probs = model.predict_proba(input_data)[:, 1]

        # print(y_probs)
        # Plot the sigmoid curve and actual data
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_probs, color='red', label='Logistic regression curve')
        plt.scatter(x[feature], y, alpha=0.2, label='Training data')
        plt.xlabel(feature)
        plt.ylabel(f'Predicted Probability of {target} = 1')
        plt.title(f'Effect of "{feature}" on {target}')
        plt.legend()
        plt.grid(True)
        plt.show()
        
def model_probability(model, X, y, target='target', positive_class=1):
    """
    Plots predicted probabilities from the full logistic regression model (using all features).
    This shows the model's sigmoid output across actual samples.
    """
    # Get predicted probabilities using all features
    y_probs = model.predict_proba(X)[:, positive_class]

    sorted_indices = np.argsort(y_probs)
    sorted_probs = y_probs[sorted_indices]
    sorted_y = np.array(y)[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_probs, label='Predicted Probability', color='red')
    plt.scatter(range(len(sorted_y)), sorted_y, alpha=0.3, label='True Labels')
    plt.title('Model Output Probability (Using All Features)')
    plt.xlabel('Sample Index (Sorted by Probability)')
    plt.ylabel(f'Predicted Probability of {target} = 1')
    plt.legend()
    plt.grid(True)
    plt.show()

'''def model_sigmoid(model, X):
    # Compute the linear combination (logit z)
    z = model.decision_function(X)

    # Apply sigmoid to get probabilities
    probs = 1 / (1 + np.exp(-z))

    # Sort by z for a smooth curve
    sorted_indices = np.argsort(z)
    z_sorted = z[sorted_indices]
    probs_sorted = probs[sorted_indices]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(z_sorted, probs_sorted, color='red', label='Sigmoid Curve')
    plt.xlabel('Combined Feature Score')
    plt.ylabel('Predicted Probability (sigmoid(z))')
    plt.title('Sigmoid Function Applied to Linear Combination of All Features')
    plt.grid(True)
    plt.legend()
    plt.show()'''
    
def model_sigmoid(model, X, y, target_name='target'):
    # Get the decision function (z = w·x + b)
    z = model.decision_function(X)
    
    # Apply sigmoid to get predicted probabilities
    probs = 1 / (1 + np.exp(-z))

    # Sort for smooth curve
    sorted_idx = np.argsort(z)
    z_sorted = z[sorted_idx]
    probs_sorted = probs[sorted_idx]
    y_sorted = y.iloc[sorted_idx]

    z_full = np.linspace(z.min() - 2, z.max() + 2, 500)
    sigmoid_curve = 1 / (1 + np.exp(-z_full))

    # Plot
    
    plt.figure(figsize=(10, 6))
    plt.plot(z_full, sigmoid_curve, color='red', label='Sigmoid (Predicted Probability)')
    plt.scatter(z_sorted, y_sorted, alpha=0.2, color='blue', label='True Labels')
    plt.xlabel('Linear Score (z = w·x + b)')
    plt.ylabel(f'Predicted Probability of {target_name} = 1')
    plt.title('Sigmoid Curve from All Features')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def performance_report(all_reports, y_train, y_pred_train, y_test, y_pred_test, name, target_parameter = 'all'):
    metric_funcs = {
        'precision': precision_score,
        'recall': recall_score,
        'accuracy': accuracy_score,
        'f1_score': f1_score
    }
    
    if target_parameter == 'all' :
        score_reports = {
            'train - precision' : precision_score(y_train, y_pred_train),
            'train - recall' : recall_score(y_train, y_pred_train),
            'train - accuracy' : accuracy_score(y_train, y_pred_train),
            'train - f1_score' : f1_score(y_train, y_pred_train),
            'test - precision' : precision_score(y_test, y_pred_test),
            'test - recall' : recall_score(y_test, y_pred_test),
            'test - accuracy_score' : accuracy_score(y_test, y_pred_test),
            'test - f1_score' : f1_score(y_test, y_pred_test),
        }
        
    else :
        if target_parameter not in metric_funcs:
            raise ValueError(f"Invalid target_parameter '{target_parameter}'. Choose from: {list(metric_funcs.keys())}")
        
        func = metric_funcs[target_parameter]
        score_reports = {
            f'train - {target_parameter}': func(y_train, y_pred_train),
            f'test - {target_parameter}': func(y_test, y_pred_test)
        }
        
    all_reports[name] = score_reports
    return all_reports

def cross_val_report(all_reports, pipeline, x_train, y_train, name, target_parameter='accuracy'):
    metric_funcs = {
        'precision': 'precision',
        'recall': 'recall',
        'accuracy': 'accuracy',
        'f1': 'f1'
        }
    
    if target_parameter not in metric_funcs:
        raise ValueError(f"Invalid target_parameter '{target_parameter}'. Choose from: {list(metric_funcs.keys())}")
    
    cross_val_pipe = cross_val_score(pipeline, 
                                x_train,
                                y_train,
                                scoring=target_parameter) # type: ignore
    
    score_reports = {
        # f'{target_parameter} -  All - Cross Validation': cross_val_pipe,
        f'{target_parameter} -  Mean - Cross Validation': cross_val_pipe.mean(),
        f'{target_parameter} -  Std - Cross Validation': cross_val_pipe.std(),
        f'{target_parameter} -  Range - Cross Validation': f"{cross_val_pipe.mean() - cross_val_pipe.std():.4f} - {cross_val_pipe.mean() + cross_val_pipe.std():.4f}"
    }
    
    all_reports[name] = score_reports
    return all_reports

def plot_pca(trained_pca, threshold=95):
    """
    Plots cumulative explained variance of PCA and marks the number of components 
    needed to reach the given variance threshold (default 95%).
    
    Parameters:
    - trained_pca: fitted sklearn.decomposition.PCA object
    - threshold: float, percentage of variance to reach (default 95)
    """
    pca = trained_pca
    cumulative_variance = np.cumsum(trained_pca.explained_variance_ratio_) * 100
    components = np.arange(1, len(cumulative_variance) + 1)

    n_components = next(i for i, total in enumerate(cumulative_variance, 1) if total >= threshold)
    
    plt.figure(figsize=[6, 4])
    plt.grid(True)
    plt.plot(components, cumulative_variance, marker='o')
    plt.axvline(x=n_components, color='red', linestyle='--', label=f'{n_components} components')
    plt.axhline(y=threshold, color='green', linestyle='--', label=f'{threshold}% variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.title(f'PCA Variance (Threshold = {threshold}%)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_pca_eigenvalue(trained_pca, threshold=0):
    """
    Plots cumulative explained variance of PCA and marks the number of components 
    needed to reach the given variance threshold (default 95%).
    
    Parameters:
    - trained_pca: fitted sklearn.decomposition.PCA object
    - threshold: float, percentage of variance to reach (default 95)
    """
    eigenvalues = trained_pca.explained_variance_
    components = np.arange(1, len(eigenvalues) + 1)
    count_above_thresh = np.sum(eigenvalues >= threshold)
    
    plt.figure(figsize=[6, 4])
    plt.grid(True)
    plt.plot(components, eigenvalues, marker='o')
    plt.axhline(y=threshold, color='green', linestyle='--', label=f'Eigenvalue = {threshold}')
    plt.axvline(x=count_above_thresh, color='red', linestyle='--', label=f'{count_above_thresh} comps ≥ {threshold}')
    plt.xlabel('Number of Components')
    plt.ylabel('Eigenvalues')
    plt.title('Eigenvalue Visualization')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def calculate_silhouette(data, minimum, maximum_cluster, randomstate=68):
    if minimum < 2:
        minimum = 2
    results = []
    param_grid = {'n_clusters': range(minimum, maximum_cluster + 1)}
    n_iteration = maximum_cluster - minimum + 1

    for params in ParameterSampler(param_grid, n_iter=n_iteration, random_state=randomstate):
        model = KMeans(n_clusters=params['n_clusters'], random_state=randomstate)
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels)
        results.append({'n_clusters': params['n_clusters'], 'silhouette_score': score})
    
    return pd.DataFrame(results).sort_values('n_clusters').reset_index(drop=True)

def plot_silhouette_scores(df_scores):
    """
    Plots silhouette scores vs number of clusters.
    
    Parameters:
    - df_scores: DataFrame with 'n_clusters' and 'silhouette_score' columns
    """
    plt.figure(figsize=(6, 4))
    plt.plot(df_scores['n_clusters'], df_scores['silhouette_score'], marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores vs. Cluster Count')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_silhouette_visualizers(data, minimum, maximum_cluster, random_state=68):
    """
    Creates a grid of SilhouetteVisualizers for a range of k values.

    Parameters:
    - data: np.array or DataFrame (must be scaled/preprocessed)
    - k_range: iterable of integers for number of clusters (default range(2, 11))
    - random_state: random seed for reproducibility
    """
    k_range = range(minimum, maximum_cluster + 1)
    n_plots = len(k_range)
    n_cols = 3
    n_rows = -(-n_plots // n_cols)  # ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    axes = axes.flatten()
    
    for idx, k in enumerate(k_range):
        ax = axes[idx]
        model = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=random_state)
        visualizer = SilhouetteVisualizer(model, colors='yellowbrick', ax=ax)
        visualizer.fit(data)
        ax.set_title(f'K = {k}')
    
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()
    
def elbow_method(data, max_cluster, random_state = 68):
    wcss = []
    for i in range(2, max_cluster+1):
        km = KMeans(n_clusters = i,
                    init = 'k-means++',
                    max_iter = 300,
                    n_init = 10,
                    random_state = random_state)
        km.fit(data)
        wcss.append(km.inertia_)

    plt.plot(range(2, max_cluster+1), wcss, marker ="o")
    plt.grid()
    plt.title('Elbow Method', fontsize = 20)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.show()
    
# Stolen from ipynb    
def plot_silhouette(range_n_clusters, X, random_state):
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 4)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters = n_clusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = random_state)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        # print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i] # type: ignore
            ith_cluster_silhouette_values.sort() # type: ignore

            size_cluster_i = ith_cluster_silhouette_values.shape[0] # type: ignore
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters) # type: ignore
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters) # type: ignore
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_

        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
        
from scipy.stats import f_oneway
import pandas as pd

def anova_feature_correlation(df, target):
    """
    Computes ANOVA F-statistics and p-values to assess correlation
    between each numerical feature and the cluster label.

    Parameters:
    - df: DataFrame with numerical features and a cluster label column
    - target: name of the cluster label column (default 'CLUSTER')

    Returns:
    - DataFrame sorted by F-statistic descending
    """
    results = []
    features = [col for col in df.columns if col != target]
    
    for feature in features:
        groups = [group[feature].values for name, group in df.groupby(target)]
        f_stat, p_val = f_oneway(*groups)
        interpretation = (
            "Significant"
            if p_val < 0.05
            else "Not Significant"
        )
        results.append({'feature': feature, 'F_statistic': f_stat, 'p_value': p_val, 'interpretation': interpretation})
    
    return pd.DataFrame(results)
