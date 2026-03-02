"""
region_visualizer.py - Standalone Analysis & Plotting Tool

Author: [Your Name]
Version: 1.1.0 (Excel Support + Enhanced Plots)

Features:
- Loads Summary_Results from CSV or Excel (.xlsx).
- Auto-parses stringified lists/dicts.
- Visualization: Pie charts, Bar charts, Boxplots.
- Filtering: Subset data by Soma Region or Neuron Type.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np
import os

class NeuronSetVisualizer:
    def __init__(self, data_source):
        """
        Args:
            data_source: Can be a pandas DataFrame OR a path to a CSV/Excel file.
        """
        # 1. Load Data
        if isinstance(data_source, str):
            print(f"Loading data from {data_source}...")
            if data_source.endswith('.xlsx'):
                # Requires 'openpyxl' installed
                self.df = pd.read_excel(data_source)
            else:
                self.df = pd.read_csv(data_source)
            
            self._parse_columns() # Fix stringified lists/dicts
            
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy()
        else:
            raise ValueError("Input must be a CSV/Excel path or pandas DataFrame")

        # 2. Standard Color Palette for Consistency
        self.type_colors = {
            'PT': '#d62728',  # Red
            'CT': '#2ca02c',  # Green
            'ITc': '#9467bd', # Purple
            'ITs': '#e377c2', # Pink
            'ITi': '#17becf', # Cyan
            'Unclassified': '#7f7f7f'
        }

    def _parse_columns(self):
        """
        Converts string representations of lists/dicts back to objects 
        (necessary if loading from CSV/Excel text).
        """
        cols_to_fix = ['Terminal_Regions', 'Region_projection_length', 'Outlier_Details']
        for col in cols_to_fix:
            if col in self.df.columns:
                # Check first row to see if it's a string
                first_val = self.df[col].iloc[0] if not self.df.empty else None
                if isinstance(first_val, str):
                    try:
                        self.df[col] = self.df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
                    except Exception as e:
                        print(f"Warning: Could not parse column {col}: {e}")

    # ==========================================
    # SUBSETTING / FILTERING TOOLS
    # ==========================================
    def filter_by_soma_keyword(self, keyword):
        """Returns a NEW NeuronSetVisualizer containing only matching somas."""
        subset = self.df[self.df['Soma_Region'].str.contains(keyword, case=False, na=False)]
        print(f"Filter '{keyword}': Reduced {len(self.df)} -> {len(subset)} neurons.")
        return NeuronSetVisualizer(subset)

    def filter_by_type(self, neuron_type):
        """Returns a NEW NeuronSetVisualizer for specific type (e.g., 'PT')."""
        subset = self.df[self.df['Neuron_Type'] == neuron_type]
        print(f"Filter Type '{neuron_type}': Reduced {len(self.df)} -> {len(subset)} neurons.")
        return NeuronSetVisualizer(subset)

    def get_data(self):
        """Returns the underlying DataFrame."""
        return self.df

    # ==========================================
    # PLOTTING FUNCTIONS
    # ==========================================
    def plot_type_distribution(self):
        """Pie/Donut chart of Neuron Types."""
        if self.df.empty: 
            print("No data to plot."); return

        counts = self.df['Neuron_Type'].value_counts()
        
        # Ensure consistent colors even if some types are missing
        colors = [self.type_colors.get(x, '#333333') for x in counts.index]

        plt.figure(figsize=(8, 8))
        
        # Enhanced Pie Plot (Donut Style)
        wedges, texts, autotexts = plt.pie(
            counts, 
            labels=counts.index, 
            autopct= lambda pct: f'{pct:.1f}%\n({int(pct/100.*counts.sum())})',  # Shows % and count
            startangle=140, 
            colors=colors, 
            pctdistance=0.85, 
            explode=[0.02]*len(counts)
        )
        
        # Donut hole
        plt.gca().add_artist(plt.Circle((0,0),0.70,fc='white'))
        plt.title(f"Neuron Type Distribution (N={len(self.df)})", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_soma_distribution(self, top_n=20):
        """Bar chart of Soma Regions."""
        if self.df.empty: return
        
        counts = self.df['Soma_Region'].value_counts().head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=counts.index, y=counts.values, palette='mako', ax=ax)
        
        # Add labels on bars
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
            
        plt.title(f'Top {top_n} Soma Regions')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_terminal_distribution(self, top_n=20):
        """Bar chart of Terminal Regions (Exploded)."""
        if self.df.empty: return

        # Explode list column to count individual targets
        exploded = self.df.explode('Terminal_Regions')
        counts = exploded['Terminal_Regions'].value_counts().head(top_n)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=counts.index, y=counts.values, palette='viridis', ax=ax)
        
        # Add labels
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', fontsize=9)

        plt.title(f"Top {top_n} Projection Targets")
        plt.ylabel("Number of Neurons Projecting Here")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_projection_length_boxplot(self):
        """Boxplot of Total Length by Soma Region."""
        if self.df.empty: return
        
        # Filter regions with enough data points (e.g., > 3 neurons)
        counts = self.df['Soma_Region'].value_counts()
        keep_regions = counts[counts > 3].index
        subset = self.df[self.df['Soma_Region'].isin(keep_regions)]

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=subset, x='Soma_Region', y='Total_Length', palette='Set3')
        plt.title('Total Axon Length by Soma Region (Min 3 neurons)')
        plt.ylabel('Length (µm)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def inspect_neuron(self, neuron_id):
        """Detailed projection report for a single neuron."""
        row = self.df[self.df['NeuronID'] == neuron_id]
        if row.empty:
            print(f"Neuron {neuron_id} not found in this set.")
            return
        
        row = row.iloc[0]
        print(f"\n--- REPORT: {row['NeuronID']} ---")
        print(f"Type:   {row['Neuron_Type']}")
        print(f"Soma:   {row['Soma_Region']}")
        print(f"Length: {row['Total_Length']:.2f} µm")
        print(f"Terminals: {len(row['Terminal_Regions'])}")

        # Projection Breakdown
        proj_dict = row['Region_projection_length']
        if not proj_dict:
            print("No projection data available.")
            return

        # Convert to Series for plotting
        stats = pd.Series(proj_dict).sort_values(ascending=False)
        stats = stats[stats > 0] # Remove zero length regions
        
        if stats.empty:
            print("No non-zero projections found.")
            return

        # Plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar Plot
        top_10 = stats.head(10)
        sns.barplot(x=top_10.values, y=top_10.index, ax=axes[0], palette='magma')
        axes[0].set_title("Top 10 Projection Targets (Length)")
        axes[0].set_xlabel("Length (µm)")

        # Pie Plot (Major vs Minor)
        if len(stats) > 6:
            main = stats.head(6)
            other = pd.Series({'Others': stats.iloc[6:].sum()})
            pie_data = pd.concat([main, other])
        else:
            pie_data = stats
            
        axes[1].pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
        axes[1].set_title("Projection Distribution")
        
        plt.tight_layout()
        plt.show()

    def get_region_matrix(self):
        """Returns the Region x Neuron matrix (useful for clustering)."""
        if self.df.empty: return pd.DataFrame()
        
        # Extract dicts
        dict_list = self.df['Region_projection_length'].tolist()
        matrix = pd.DataFrame(dict_list)
        matrix.fillna(0, inplace=True)
        
        # Add Metadata
        matrix.insert(0, 'NeuronID', self.df['NeuronID'])
        matrix.insert(1, 'Neuron_Type', self.df['Neuron_Type'])
        matrix.insert(2, 'Soma_Region', self.df['Soma_Region'])
        
        return matrix

# ==========================================
# SAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # Define your input file (Example: Summary_Results.xlsx or .csv)
    INPUT_FILE = 'INS_df_v2.xlsx' 
    
    # Check if file exists to prevent errors
    if not os.path.exists(INPUT_FILE):
        print(f"Sample file '{INPUT_FILE}' not found. Please generate it using region_analysis.py first.")
    else:
        # 1. Initialize Visualizer
        viz = NeuronSetVisualizer(INPUT_FILE)
        print("Data Loaded Successfully.")

        # 2. Global Overview Plots
        print("Displaying Type Distribution...")
        viz.plot_type_distribution()
        
        print("Displaying Soma Distribution...")
        viz.plot_soma_distribution(top_n=15)

        # # 3. Filtering Example: Analyze only PT Neurons
        # print("\n--- Analyzing PT Subset ---")
        # pt_viz = viz.filter_by_type('PT')
        
        # if not pt_viz.get_data().empty:
        #     pt_viz.plot_projection_length_boxplot()
        #     pt_viz.plot_terminal_distribution(top_n=10)
        
        # # 4. Filtering Example: Analyze Cingulate Cortex (Area 24)
        # print("\n--- Analyzing Cingulate Subset ---")
        # cingulate_viz = viz.filter_by_soma_keyword('area_24')
        
        # if not cingulate_viz.get_data().empty:
        #     cingulate_viz.plot_type_distribution()
            
        # # 5. Single Neuron Inspection
        # # Grab the ID of the first neuron in the list
        # first_neuron_id = viz.get_data()['NeuronID'].iloc[0]
        # viz.inspect_neuron(first_neuron_id)