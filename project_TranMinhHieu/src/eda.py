"""
Exploratory Data Analysis Module
Performs statistical analysis and generates visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def set_plot_style():
    """Set the default plotting style."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def basic_statistics(df):
    """
    Display basic statistics of the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("="*50)
    print("BASIC STATISTICS")
    print("="*50)
    
    print("\nDataset Shape:", df.shape)
    print("\nColumn Names and Types:")
    print(df.dtypes)
    
    print("\nNumerical Statistics:")
    print(df.describe())
    
    print("\nCategorical Statistics:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'date':
            print(f"\n{col} - Value Counts:")
            print(df[col].value_counts())


def plot_disaster_types(df, save_fig=False):
    """
    Plot distribution of disaster types.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    save_fig : bool
        Whether to save the figure
    """
    set_plot_style()
    
    plt.figure(figsize=(12, 6))
    disaster_counts = df['disaster_type'].value_counts()
    
    ax = sns.barplot(x=disaster_counts.index, y=disaster_counts.values, palette='viridis')
    plt.title('Distribution of Disaster Types', fontsize=16, fontweight='bold')
    plt.xlabel('Disaster Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(disaster_counts.values):
        ax.text(i, v + 0.2, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    if save_fig:
        plt.savefig('outputs/disaster_types_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_temporal_trends(df, save_fig=False):
    """
    Plot temporal trends of disasters.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with date column
    save_fig : bool
        Whether to save the figure
    """
    set_plot_style()
    
    if 'date' not in df.columns:
        print("Date column not found in dataframe")
        return
    
    df_temp = df.copy()
    df_temp['year'] = pd.to_datetime(df_temp['date']).dt.year
    
    plt.figure(figsize=(14, 6))
    yearly_counts = df_temp.groupby('year').size()
    
    plt.subplot(1, 2, 1)
    plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=8)
    plt.title('Number of Disasters by Year', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Disasters', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    disaster_year = df_temp.groupby(['year', 'disaster_type']).size().unstack(fill_value=0)
    disaster_year.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='tab10')
    plt.title('Disaster Types by Year', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Disaster Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig('outputs/temporal_trends.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_geographical_distribution(df, save_fig=False):
    """
    Plot geographical distribution of disasters.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    save_fig : bool
        Whether to save the figure
    """
    set_plot_style()
    
    plt.figure(figsize=(14, 8))
    
    # By region
    plt.subplot(2, 1, 1)
    region_counts = df['region'].value_counts()
    sns.barplot(x=region_counts.index, y=region_counts.values, palette='coolwarm')
    plt.title('Distribution of Disasters by Region', fontsize=14, fontweight='bold')
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    
    # Top countries
    plt.subplot(2, 1, 2)
    country_counts = df['country'].value_counts().head(10)
    sns.barplot(x=country_counts.values, y=country_counts.index, palette='rocket')
    plt.title('Top 10 Countries by Number of Disasters', fontsize=14, fontweight='bold')
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig('outputs/geographical_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_impact_analysis(df, save_fig=False):
    """
    Plot impact analysis (casualties, affected population, economic impact).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    save_fig : bool
        Whether to save the figure
    """
    set_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Casualties by disaster type
    casualties_by_type = df.groupby('disaster_type')['casualties'].sum().sort_values(ascending=False)
    axes[0, 0].barh(casualties_by_type.index, casualties_by_type.values, color='crimson')
    axes[0, 0].set_title('Total Casualties by Disaster Type', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Total Casualties')
    
    # Affected population by disaster type
    affected_by_type = df.groupby('disaster_type')['affected_population'].sum().sort_values(ascending=False)
    axes[0, 1].barh(affected_by_type.index, affected_by_type.values, color='orange')
    axes[0, 1].set_title('Total Affected Population by Disaster Type', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Total Affected Population')
    
    # Economic impact by disaster type
    economic_by_type = df.groupby('disaster_type')['economic_impact_usd'].sum().sort_values(ascending=False)
    axes[1, 0].barh(economic_by_type.index, economic_by_type.values / 1e9, color='green')
    axes[1, 0].set_title('Total Economic Impact by Disaster Type', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Economic Impact (Billion USD)')
    
    # Response effectiveness by disaster type
    response_by_type = df.groupby('disaster_type')['response_effectiveness'].mean().sort_values(ascending=False)
    axes[1, 1].barh(response_by_type.index, response_by_type.values, color='steelblue')
    axes[1, 1].set_title('Average Response Effectiveness by Disaster Type', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Average Response Effectiveness')
    axes[1, 1].set_xlim(0, 1)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig('outputs/impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_correlation_matrix(df, save_fig=False):
    """
    Plot correlation matrix of numerical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    save_fig : bool
        Whether to save the figure
    """
    set_plot_style()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
    plt.title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('outputs/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def perform_eda(df, save_figs=False):
    """
    Perform complete exploratory data analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    save_figs : bool
        Whether to save figures
    """
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50 + "\n")
    
    # Basic statistics
    basic_statistics(df)
    
    # Create outputs directory if saving figures
    if save_figs:
        Path('outputs').mkdir(exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_disaster_types(df, save_figs)
    plot_temporal_trends(df, save_figs)
    plot_geographical_distribution(df, save_figs)
    plot_impact_analysis(df, save_figs)
    plot_correlation_matrix(df, save_figs)
    
    print("\nEDA complete!")


if __name__ == "__main__":
    # Load preprocessed data
    from preprocessing import load_data, preprocess_data
    
    df = load_data('data/global_disaster_response_2018_2024.csv')
    if df is not None:
        df = preprocess_data('data/global_disaster_response_2018_2024.csv', save_output=False)
        perform_eda(df, save_figs=True)
