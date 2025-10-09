""""
Mix of functions necessary for the sprint1-visualizations noteboook
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency, kruskal, f_oneway


def plot_rates(data, target, categorical_vars, n_cols):
    """
    Creates stacked barplots of H1N1 and seasonal vaccination rates
    for each categorical variable in `categorical_vars`.
    """
    custom_orders = {
        'age_group': ['18 - 34 Years', '35 - 44 Years', '45 - 54 Years', '55 - 64 Years', '65+ Years'],
        'education': ['< 12 Years', '12 Years', 'Some College', 'College Graduate'],
        'income_poverty': ['Below Poverty', '<= $75,000, Above Poverty', '> $75,000'],
        'household_adults': [0.0, 1.0, 2.0, 3.0],
        'household_children': [0.0, 1.0, 2.0, 3.0],
    }
    
    n_vars = len(categorical_vars)
    nrows = int(np.ceil(n_vars / n_cols))

    fig, axes = plt.subplots(nrows=nrows, ncols=n_cols, figsize=(12, nrows * 4))
    axes = axes.flatten()

    for i, (var, title) in enumerate(categorical_vars.items()):
        ax = axes[i]

        # Aggregate statistics
        stats = data.groupby(var).agg({
            'h1n1_vaccine': ['mean', 'count'],
            'seasonal_vaccine': 'mean'
        }).round(4)
        stats.columns = ['h1n1_rate', 'count', 'seasonal_rate']
        stats = stats[stats['count'] > 50]  # filter small groups

        # Apply custom sorting if available, otherwise sort by h1n1_rate
        if var in custom_orders:
            # Filter to only include categories that exist in the data
            order = [cat for cat in custom_orders[var] if cat in stats.index]
            stats = stats.reindex(order)
        else:
            stats = stats.sort_values('h1n1_rate', ascending=True)
        
        stats['h1n1_rate'] *= 100
        stats['seasonal_rate'] *= 100

        # Stacked bar chart
        y_pos = np.arange(len(stats))
        ax.barh(y_pos, stats['h1n1_rate'], color='#FF6B6B', alpha=0.8, label='H1N1')
        ax.barh(y_pos, stats['seasonal_rate'], 
                left=stats['h1n1_rate'], color='#4ECDC4', alpha=0.8, label='Seasonal')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(stats.index, fontsize=9)
        ax.set_xlabel('Vaccination Rate (%)', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Optional: add value labels
        for j, (h1n1, seasonal) in enumerate(zip(stats['h1n1_rate'], stats['seasonal_rate'])):
            ax.text(h1n1 / 2, j, f'{h1n1:.1f}%', color='white', ha='center', va='center', fontsize=8)
            ax.text(h1n1 + seasonal / 2, j, f'{seasonal:.1f}%', color='white', ha='center', va='center', fontsize=8)

    # Clean up extra axes
    for k in range(i + 1, len(axes)):
        fig.delaxes(axes[k])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=10)
    fig.suptitle(f"Vaccination Rates by {target} Variables", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()


def plot_heatmap(data, target, categorical_vars, n_cols):
    """
    Creates stacked heatmaps for H1N1 and Seasonal vaccine rates per each categorical variables 
    """
    fig, axes = plt.subplots(nrows=len(categorical_vars)//2 + 1, ncols=n_cols, figsize=(12, len(categorical_vars)*2))
    axes = axes.flatten()

    for i, (var, title) in enumerate(categorical_vars.items()):
        demo_stats = data.groupby(var).agg({
            'h1n1_vaccine': 'mean',
            'seasonal_vaccine': 'mean'
        }).round(3) * 100

        sns.heatmap(
            demo_stats.T, 
            annot=True, fmt=".1f", cmap="coolwarm", cbar=False, 
            ax=axes[i], linewidths=0.5, annot_kws={"size": 9}
        )

        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].set_yticklabels(['H1N1', 'Seasonal'], fontsize=10)


    for k in range(i + 1, len(axes)):
        fig.delaxes(axes[k])

    fig.suptitle(f"Vaccination Rate Heatmaps by {target} Variable", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# Helper: Cramér’s V (effect size for categorical associations)
def cramers_v(confusion_matrix):
    """Compute Cramér’s V for categorical association strength"""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

# Analysis and stats tests function
def analyse_test(data, target, categorical_vars, tests=['chi2'], alpha=0.05, visualize=True):
    """
    Runs statistical tests for categorical variables vs. vaccination outcomes and visualizes results.

    Parameters:
        data (pd.DataFrame): dataset with categorical vars + vaccination flags
        target: String of target variable 
        categorical_vars (dict or list): variables to test
        tests (list): any combination of ['chi2', 'cramers_v', 'anova', 'kruskal']
        alpha (float): significance threshold
        visualize (bool): whether to plot summary visualization
    
    Returns:
        results_df (pd.DataFrame): table of test statistics, p-values, and significance flags
  
    """
    test_results = []

    for var in categorical_vars:
        if data[var].nunique() < 2:
            continue  # skip single-category variables

        print(f"Variable: {var}")
        for vaccine in ['h1n1_vaccine', 'seasonal_vaccine']:
            print(f"Vaccine: {vaccine.replace('_vaccine', '').upper()}")
            #Chi2 test
            if 'chi2' in tests:
                contingency = pd.crosstab(data[var], data[vaccine])
                chi2, p, dof, expected = chi2_contingency(contingency)
                sig = "Significant" if p < alpha else "Not significant"
                print(f"    • Chi-square test: χ²={chi2:.2f}, p={p:.4g} → {sig}")
                test_results.append({
                    'Variable': var, 
                    'Vaccine': vaccine,
                    'Test': 'Chi-square', 
                    'Statistic': chi2,
                    'p_value': p, 
                    'Effect': np.nan
                })

            # ---------- Cramér’s V ----------
            if 'cramers_v' in tests:
                contingency = pd.crosstab(data[var], data[vaccine])
                effect = cramers_v(contingency)
                strength = (
                    "weak" if effect < 0.1 else
                    "moderate" if effect < 0.3 else
                    "strong"
                )
                print(f"    • Cramér’s V: {effect:.3f} ({strength} association)")
                test_results.append({
                    'Variable': var, 
                    'Vaccine': vaccine,
                    'Test': 'Cramers V', 
                    'Statistic': np.nan,
                    'p_value': np.nan, 
                    'Effect': effect
                })

            # ---------- ANOVA ----------
            if 'anova' in tests and pd.api.types.is_numeric_dtype(data[var]):
                groups = [g[vaccine].values for _, g in data.groupby(var)]
                f_stat, p = f_oneway(*groups)
                sig = "Significant" if p < alpha else "Not significant"
                print(f"    • ANOVA: F={f_stat:.2f}, p={p:.4g} → {sig}")
                test_results.append({
                    'Variable': var, 
                    'Vaccine': vaccine,
                    'Test': 'ANOVA', 
                    'Statistic': f_stat,
                    'p_value': p, 
                    'Effect': np.nan
                })

            # ---------- Kruskal–Wallis ----------
            if 'kruskal' in tests and pd.api.types.is_numeric_dtype(data[var]):
                groups = [g[vaccine].values for _, g in data.groupby(var)]
                stat, p = kruskal(*groups)
                sig = "Significant" if p < alpha else "Not significant"
                print(f"    • Kruskal–Wallis: H={stat:.2f}, p={p:.4g} → {sig}")
                test_results.append({
                    'Variable': var, 'Vaccine': vaccine,
                    'Test': 'Kruskal–Wallis', 'Statistic': stat,
                    'p_value': p, 'Effect': np.nan
                })

    results_df = pd.DataFrame(test_results)
    if results_df.empty:
        print("No valid results — check your inputs.")
        return pd.DataFrame()

    results_df['Significant'] = results_df['p_value'] < alpha

    # Visualization: combine significance & effect size
    if visualize:
        plot_df = results_df.pivot_table(
            index='Variable', columns='Vaccine', values='p_value'
        ).applymap(lambda x: -np.log10(x) if pd.notna(x) else np.nan)
        effect_df = results_df.query("Test == 'Cramers V'").pivot_table(
            index='Variable', columns='Vaccine', values='Effect'
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(categorical_vars) * 0.5)))

        # --- Left: significance (–log10 p) with capped color scale ---
        sns.heatmap(plot_df, annot=True, fmt=".2f", cmap="Reds",
                    vmin=0, vmax=50,
                    cbar_kws={'label': '–log₁₀(p)'}, ax=axes[0])
        axes[0].set_title('Significance of Association', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('')
        axes[0].set_xlabel('Vaccine Type')

        # --- Right: effect size (Cramér's V) ---
        sns.heatmap(effect_df, annot=True, fmt=".2f", cmap="Blues",
                    vmin=0, vmax=0.5,
                    cbar_kws={'label': "Cramér's V"}, ax=axes[1])
        axes[1].set_title('Effect Size', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('')
        axes[1].set_xlabel('Vaccine Type')


        plt.suptitle(f'Statistical Tests: {target} Effects on Vaccination Rates',
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    print(f"\n {target} analysis complete. Summary table returned.\n")
    return results_df

def analyse_and_rank(data, targets=['h1n1_vaccine', 'seasonal_vaccine'], alpha=0.05, visualize=True):
    """
    Run statistical significance tests for *all* features vs. both vaccine targets.
    Automatically selects appropriate tests for each variable type.

    Returns a ranked DataFrame by p-value or effect size.
    """

    results = []

    for var in data.columns:
        if data[var].nunique() < 2:
            continue  # skip constant columns

        for vaccine in targets:
            target = data[vaccine]

            # Categorical features → Chi-square + Cramér’s V
            if not pd.api.types.is_numeric_dtype(data[var]) or data[var].nunique() <= 10:
                contingency = pd.crosstab(data[var], target)
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                chi2, p, dof, expected = chi2_contingency(contingency)
                effect = cramers_v(contingency)

                results.append({
                    'Feature': var,
                    'Vaccine': vaccine,
                    'Test': 'Chi-square',
                    'Statistic': chi2,
                    'p_value': p,
                    'Effect_Size': effect
                })

            # Continuous or ordinal features → ANOVA/Kruskal
            else:
                groups = [data.loc[target == t, var] for t in target.unique() if len(data.loc[target == t, var]) > 1]
                if len(groups) < 2:
                    continue
                f_stat, p = f_oneway(*groups)
                results.append({
                    'Feature': var,
                    'Vaccine': vaccine,
                    'Test': 'ANOVA',
                    'Statistic': f_stat,
                    'p_value': p,
                    'Effect_Size': np.nan
                })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No valid results found. Check dataset or target names.")
        return pd.DataFrame()

    # Rank features by significance (lower p-value → higher rank)
    results_df['Rank'] = results_df.groupby('Vaccine')['p_value'].rank(method='dense', ascending=True)

    # Add significance flag
    results_df['Significant'] = results_df['p_value'] < alpha

    # Sort
    results_df = results_df.sort_values(['Vaccine', 'p_value'])

    # Optional visualization
    if visualize:
        plt.figure(figsize=(10, max(6, len(results_df['Feature'].unique()) * 0.25)))
        sns.barplot(data=results_df, y='Feature', x='-log10(p)', hue='Vaccine',
                    orient='h', palette='Set2')
        plt.title('Top Features by Statistical Significance')
        plt.xlabel('-log10(p-value)')
        plt.tight_layout()
        plt.show()

    return results_df