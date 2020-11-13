import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from clean import *

def hist_plot(df):
    df.hist(figsize=(12,8))
    plt.tight_layout()

def comparison_bar_plot(x, y, df, color, title, xlabel, ylabel, ax):
    ax = sns.barplot(x=x, y=y, data=df, color=color, alpha=0.5)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.tight_layout()
        
def image_of_plot(file):
    '''Input: file - file path for image (string)
        Creates a png file of a plot and saves it to images folder
        Output: png file'''
    return plt.savefig(file, transparent=False, 
    bbox_inches='tight', format='svg', dpi=1200)



if __name__ == '__main__':
    file = (input("Enter path of filename: "))
    df = clean_this_df(file)
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_bar_plot(x='avg_rating_of_driver', y='churn', df=df, 
    color='m', title='Rider Churn Rate Compared to Avg Rating of Driver', 
    xlabel='Average Rating of Driver', ylabel='Churn Rate', ax=ax)
    image_of_plot('../images/driver_rate_churn_compare.svg')
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_bar_plot(x='weekday_pct', y='churn', df=df, 
    color='m', title='Rider Churn Rate Compared to Weekday Percentage', 
    xlabel='Weekday Percentage', ylabel='Churn Rate', ax=ax)
    image_of_plot('../images/weekday_pct_churn_compare.svg')
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_bar_plot(x='avg_dist', y='churn', df=df, 
    color='m', title='Rider Churn Rate Compared to Avg Trip Distance', 
    xlabel='Average Distance (miles)', ylabel='Churn Rate', ax=ax)
    image_of_plot('../images/avg_dist_churn_compare.svg')