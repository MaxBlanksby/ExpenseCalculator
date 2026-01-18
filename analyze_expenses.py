"""
Expense Analysis Script
Analyzes CSV files containing transaction data, tallies expenses by merchant,
and generates monthly and yearly reports with visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json
from collections import defaultdict


def read_csv_files(csv_folder='Csv'):
    """
    Read all CSV files from the specified folder.
    
    Args:
        csv_folder: Path to folder containing CSV files
        
    Returns:
        Dictionary with month names as keys and DataFrames as values
    """
    csv_path = Path(csv_folder)
    csv_files = sorted(csv_path.glob('*.csv'))
    
    data_by_month = {}
    
    for file in csv_files:
        # Extract month name from filename (e.g., "October2025_4952.csv" -> "October2025")
        month_name = file.stem.split('_')[0]
        
        try:
            df = pd.read_csv(file)
            data_by_month[month_name] = df
            print(f"Loaded {file.name}: {len(df)} transactions")
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
    
    return data_by_month


def clean_merchant_name(payee):
    """
    Clean and standardize merchant names for better grouping.
    
    Args:
        payee: Raw payee string from CSV
        
    Returns:
        Cleaned merchant name
    """
    if pd.isna(payee):
        return "Unknown"
    
    # Remove extra spaces and convert to title case
    cleaned = ' '.join(payee.split())
    
    # You can add more cleaning rules here as needed
    # For example, removing transaction IDs, standardizing names, etc.
    
    return cleaned


def analyze_monthly_expenses(df, month_name):
    """
    Analyze expenses for a single month.
    
    Args:
        df: DataFrame containing transaction data
        month_name: Name of the month
        
    Returns:
        Dictionary with merchant totals (expenses only)
    """
    # Filter only expenses (negative amounts)
    expenses = df[df['Amount'] < 0].copy()
    
    # Clean merchant names
    expenses['Merchant'] = expenses['Payee'].apply(clean_merchant_name)
    
    # Group by merchant and sum amounts
    merchant_totals = expenses.groupby('Merchant')['Amount'].sum().sort_values()
    
    # Convert to dictionary
    return merchant_totals.to_dict()


def calculate_monthly_totals(data_by_month):
    """
    Calculate total expenses for each month.
    
    Args:
        data_by_month: Dictionary of DataFrames by month
        
    Returns:
        Dictionary with month names and total expenses
    """
    monthly_totals = {}
    
    for month_name, df in data_by_month.items():
        # Only sum negative amounts (expenses)
        total_expenses = df[df['Amount'] < 0]['Amount'].sum()
        monthly_totals[month_name] = total_expenses
    
    return monthly_totals


def aggregate_yearly_expenses(data_by_month):
    """
    Aggregate all expenses across all months by merchant.
    
    Args:
        data_by_month: Dictionary of DataFrames by month
        
    Returns:
        Dictionary with merchant names and yearly totals
    """
    yearly_totals = defaultdict(float)
    
    for month_name, df in data_by_month.items():
        expenses = df[df['Amount'] < 0].copy()
        expenses['Merchant'] = expenses['Payee'].apply(clean_merchant_name)
        
        for merchant, amount in expenses.groupby('Merchant')['Amount'].sum().items():
            yearly_totals[merchant] += amount
    
    # Sort by amount spent (most negative first)
    return dict(sorted(yearly_totals.items(), key=lambda x: x[1]))


def save_monthly_reports(data_by_month, output_folder='Reports'):
    """
    Save individual monthly reports as CSV files compatible with Google Sheets.
    
    Args:
        data_by_month: Dictionary of DataFrames by month
        output_folder: Folder to save reports
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    for month_name, df in data_by_month.items():
        merchant_totals = analyze_monthly_expenses(df, month_name)
        
        # Convert to DataFrame for CSV export
        report_df = pd.DataFrame([
            {'Merchant': merchant, 'Amount': amount, 'Amount_Absolute': abs(amount)}
            for merchant, amount in sorted(merchant_totals.items(), key=lambda x: x[1])
        ])
        
        # Add total row
        total = sum(merchant_totals.values())
        total_row = pd.DataFrame([{
            'Merchant': 'TOTAL EXPENSES',
            'Amount': total,
            'Amount_Absolute': abs(total)
        }])
        report_df = pd.concat([report_df, total_row], ignore_index=True)
        
        # Save as CSV compatible with Google Sheets
        csv_file = output_path / f'{month_name}_expenses.csv'
        report_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        print(f"Saved CSV report for {month_name}")


def save_yearly_report(yearly_totals, output_folder='Reports'):
    """
    Save yearly aggregated report as CSV compatible with Google Sheets.
    
    Args:
        yearly_totals: Dictionary with merchant yearly totals
        output_folder: Folder to save report
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Convert to DataFrame for CSV export
    report_df = pd.DataFrame([
        {'Merchant': merchant, 'Amount': amount, 'Amount_Absolute': abs(amount)}
        for merchant, amount in sorted(yearly_totals.items(), key=lambda x: x[1])
    ])
    
    # Add total row
    total = sum(yearly_totals.values())
    total_row = pd.DataFrame([{
        'Merchant': 'TOTAL YEARLY EXPENSES',
        'Amount': total,
        'Amount_Absolute': abs(total)
    }])
    report_df = pd.concat([report_df, total_row], ignore_index=True)
    
    # Save as CSV compatible with Google Sheets
    csv_file = output_path / 'yearly_expenses.csv'
    report_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    print(f"\nYearly CSV report saved to {csv_file}")


def create_monthly_expense_chart(monthly_totals, output_folder='Reports'):
    """
    Create a bar chart showing total expenses for each month.
    
    Args:
        monthly_totals: Dictionary with month names and total expenses
        output_folder: Folder to save chart
    """
    # Sort months chronologically
    month_order = ['January2025', 'February2025', 'March2025', 'April2025', 
                   'May2025', 'June2025', 'July2025', 'August2025',
                   'September2025', 'October2025', 'November2025', 'December2025']
    
    # Filter to only include months that exist in the data
    sorted_months = [m for m in month_order if m in monthly_totals]
    sorted_totals = [monthly_totals[m] for m in sorted_months]
    
    # Convert to positive values for better visualization
    sorted_totals_positive = [-x for x in sorted_totals]
    
    # Create short labels (just month name)
    short_labels = [m.replace('2025', '') for m in sorted_months]
    
    # Create bar chart
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(sorted_months)), sorted_totals_positive, 
                   color='#FF6B6B', edgecolor='#C92A2A', linewidth=1.5)
    
    # Customize chart
    plt.xlabel('Month', fontsize=12, fontweight='bold')
    plt.ylabel('Total Expenses ($)', fontsize=12, fontweight='bold')
    plt.title('Monthly Expenses - 2025', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(range(len(sorted_months)), short_labels, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars
    for i, (bar, value) in enumerate(zip(bars, sorted_totals_positive)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'${value:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add average line
    avg_expense = sum(sorted_totals_positive) / len(sorted_totals_positive)
    plt.axhline(y=avg_expense, color='#2E86AB', linestyle='--', linewidth=2, 
                label=f'Average: ${avg_expense:,.0f}')
    plt.legend()
    
    plt.tight_layout()
    
    # Save chart
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    chart_file = output_path / 'monthly_expenses_chart.png'
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to {chart_file}")
    plt.show()


def create_top_merchants_chart(yearly_totals, top_n=15, output_folder='Reports'):
    """
    Create a bar chart showing top merchants by spending.
    
    Args:
        yearly_totals: Dictionary with merchant yearly totals
        top_n: Number of top merchants to display
        output_folder: Folder to save chart
    """
    # Get top N merchants by spending
    sorted_merchants = sorted(yearly_totals.items(), key=lambda x: x[1])[:top_n]
    merchants = [m[0] for m in sorted_merchants]
    amounts = [-m[1] for m in sorted_merchants]  # Convert to positive
    
    # Create horizontal bar chart
    plt.figure(figsize=(12, 10))
    bars = plt.barh(range(len(merchants)), amounts, color='#4ECDC4', 
                    edgecolor='#1A535C', linewidth=1.5)
    
    # Customize chart
    plt.xlabel('Total Spent ($)', fontsize=12, fontweight='bold')
    plt.ylabel('Merchant', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Merchants by Spending - 2025', fontsize=16, fontweight='bold', pad=20)
    plt.yticks(range(len(merchants)), merchants, fontsize=9)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, amounts)):
        plt.text(value + 20, bar.get_y() + bar.get_height()/2,
                f'${value:,.2f}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save chart
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    chart_file = output_path / 'top_merchants_chart.png'
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"Top merchants chart saved to {chart_file}")
    plt.show()


def print_summary(monthly_totals, yearly_totals):
    """
    Print a summary of the analysis to console.
    
    Args:
        monthly_totals: Dictionary with month totals
        yearly_totals: Dictionary with yearly merchant totals
    """
    print("\n" + "=" * 80)
    print("EXPENSE ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Monthly summary
    print("\nMONTHLY TOTALS:")
    print("-" * 80)
    for month, total in sorted(monthly_totals.items()):
        print(f"{month:20s} ${total:10.2f} (${-total:,.2f} spent)")
    
    total_yearly = sum(monthly_totals.values())
    print("-" * 80)
    print(f"{'TOTAL YEAR':20s} ${total_yearly:10.2f} (${-total_yearly:,.2f} spent)")
    
    # Top merchants
    print("\n\nTOP 10 MERCHANTS BY SPENDING:")
    print("-" * 80)
    top_merchants = sorted(yearly_totals.items(), key=lambda x: x[1])[:10]
    for merchant, amount in top_merchants:
        print(f"{merchant:60s} ${amount:10.2f} (${-amount:,.2f})")
    
    print("\n" + "=" * 80 + "\n")


def main():
    """
    Main function to run the complete expense analysis.
    """
    print("Starting Expense Analysis...")
    print("-" * 80)
    
    # Read all CSV files
    data_by_month = read_csv_files('Csv')
    
    if not data_by_month:
        print("No CSV files found in the 'Csv' folder!")
        return
    
    print(f"\nLoaded {len(data_by_month)} months of data")
    
    # Calculate monthly totals
    monthly_totals = calculate_monthly_totals(data_by_month)
    
    # Save individual monthly reports
    print("\nGenerating monthly reports...")
    save_monthly_reports(data_by_month)
    
    # Aggregate yearly expenses
    print("\nAggregating yearly expenses...")
    yearly_totals = aggregate_yearly_expenses(data_by_month)
    
    # Save yearly report
    save_yearly_report(yearly_totals)
    
    # Create visualizations
    print("\nGenerating charts...")
    create_monthly_expense_chart(monthly_totals)
    create_top_merchants_chart(yearly_totals)
    
    # Print summary
    print_summary(monthly_totals, yearly_totals)
    
    print("Analysis complete! Check the 'Reports' folder for detailed results.")


if __name__ == "__main__":
    main()
