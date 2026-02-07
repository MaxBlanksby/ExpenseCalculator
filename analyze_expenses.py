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


def detect_csv_format(df, filename):
    """
    Detect the CSV format based on column names.
    
    Args:
        df: DataFrame to analyze
        filename: Name of the file for context
        
    Returns:
        String indicating format: 'monthly', 'prime', 'chase', 'citi', or 'unknown'
    """
    columns = set(df.columns)
    
    if 'Posted Date' in columns and 'Payee' in columns and 'Amount' in columns:
        format_type = 'monthly'
    elif 'Transaction Date' in columns and 'Description' in columns and 'Amount' in columns:
        format_type = 'prime'
    elif 'Posting Date' in columns and 'Description' in columns and 'Amount' in columns:
        format_type = 'chase'
    elif 'Date' in columns and 'Description' in columns and 'Amount' in columns:
        format_type = 'keybank'
    elif 'Date' in columns and 'Description' in columns and ('Debit' in columns or 'Credit' in columns):
        format_type = 'citi'
    else:
        format_type = 'unknown'
    print(f"Detected CSV format for {filename}: {format_type}")
    return format_type


def normalize_csv_data(df, csv_format):
    """
    Normalize different CSV formats to a standard format with Date, Payee, Amount columns.
    
    Args:
        df: DataFrame to normalize
        csv_format: Detected format ('monthly', 'prime', 'citi')
        
    Returns:
        Normalized DataFrame with Date, Payee, Amount columns
    """
    if csv_format == 'monthly':
        normalized = pd.DataFrame({
            'Date': pd.to_datetime(df['Posted Date'], format='%m/%d/%Y'),
            'Payee': df['Payee'],
            'Amount': df['Amount']
        })
    elif csv_format == 'prime':
        normalized = pd.DataFrame({
            'Date': pd.to_datetime(df['Transaction Date'], format='%m/%d/%Y'),
            'Payee': df['Description'],
            'Amount': df['Amount']
        })
    elif csv_format == 'chase':
        normalized = pd.DataFrame({
            'Date': pd.to_datetime(df['Posting Date'], format='%m/%d/%y'),
            'Payee': df['Description'],
            'Amount': df['Amount']
        })
    elif csv_format == 'keybank':
        normalized = pd.DataFrame({
            'Date': pd.to_datetime(df['Date'], format='%m/%d/%y'),
            'Payee': df['Description'],
            'Amount': df['Amount']
        })
    elif csv_format == 'citi':
        # Citi has separate Debit/Credit columns
        # Debit = money spent (positive in file, should be negative)
        # Credit = refunds/payments (positive, should be positive or ignored)
        amounts = []
        for _, row in df.iterrows():
            debit = row.get('Debit', 0)
            credit = row.get('Credit', 0)
            # Convert to float, handling NaN
            debit = float(debit) if pd.notna(debit) else 0
            credit = float(credit) if pd.notna(credit) else 0
            # Debit is spending (make negative), Credit is refund (keep positive)
            amount = -debit + credit
            amounts.append(amount)
        
        normalized = pd.DataFrame({
            'Date': pd.to_datetime(df['Date'], format='%m/%d/%Y'),
            'Payee': df['Description'],
            'Amount': amounts
        })
    else:
        return None
    
    return normalized


def read_csv_files(csv_folder='Csv'):
    """
    Read all CSV files from the specified folder and normalize them.
    
    Args:
        csv_folder: Path to folder containing CSV files
        
    Returns:
        Dictionary with month names as keys and normalized DataFrames as values
    """
    csv_path = Path(csv_folder)
    csv_files = sorted(csv_path.glob('*.csv')) + sorted(csv_path.glob('*.CSV'))
    
    data_by_month = {}
    
    for file in csv_files:
        # Extract month name from filename (e.g., "October2025_4952.csv" -> "October2025")
        month_name = file.stem.split('_')[0]
        
        try:
            df = pd.read_csv(file)
            csv_format = detect_csv_format(df, file.name)
            
            if csv_format == 'unknown':
                print(f"Warning: Unknown format for {file.name}, skipping...")
                continue
            
            normalized = normalize_csv_data(df, csv_format)
            if normalized is not None:
                data_by_month[month_name] = normalized
                print(f"Loaded {file.name}: {len(normalized)} transactions ({csv_format} format)")
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
    
    return data_by_month


def read_all_csv_files_normalized(csv_folder='Csv'):
    """
    Read all CSV files from the specified folder and normalize them to a standard format.
    Handles multiple CSV formats (monthly statements, prime, Citi).
    
    Args:
        csv_folder: Path to folder containing CSV files
        
    Returns:
        Single DataFrame with all transactions normalized (Date, Payee, Amount)
    """
    csv_path = Path(csv_folder)
    csv_files = sorted(csv_path.glob('*.csv')) + sorted(csv_path.glob('*.CSV'))
    
    all_transactions = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            csv_format = detect_csv_format(df, file.name)
            
            if csv_format == 'unknown':
                print(f"Warning: Unknown format for {file.name}, skipping...")
                continue
            
            normalized = normalize_csv_data(df, csv_format)
            if normalized is not None:
                all_transactions.append(normalized)
                print(f"Loaded {file.name}: {len(normalized)} transactions ({csv_format} format)")
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
    
    if all_transactions:
        combined = pd.concat(all_transactions, ignore_index=True)
        # Remove duplicates based on Date, Payee, Amount
        combined = combined.drop_duplicates(subset=['Date', 'Payee', 'Amount'])
        # Sort by date
        combined = combined.sort_values('Date')
        return combined
    
    return pd.DataFrame()


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
    
    # Check if payee starts with common prefixes and standardize to common names
    if str(payee).startswith('HOP*'):
        return "HOP TRIMET"
    elif str(payee).startswith('Dropbox'):
        return "Dropbox"
    elif str(payee).startswith('GOOGLE *FI'):
        return "Google Fi"
    elif str(payee).startswith('SAFEWAY'):
        return "Safeway"
    elif str(payee).startswith('NORDSTROM'):
        return "Nordstrom"
    elif str(payee).startswith('NORDRACK'):
        return "Nordstrom Rack"
    elif str(payee).startswith('WALMART'):
        return "Walmart"
    elif str(payee).startswith('TARGET'):
        return "Target"
    elif str(payee).startswith('COSTCO'):
        return "Costco"
    elif str(payee).startswith('WHOLEFDS'):
        return "Whole Foods"
    elif str(payee).startswith('AMAZON'):
        return "Amazon"
    elif str(payee).startswith('AMZN'):
        return "Amazon"
    elif str(payee).startswith('Amazon'):
        return "Amazon"
    elif str(payee).startswith('Kindle'):
        return "Kindle"
    elif str(payee).startswith('Prime Video'):
        return "Prime Video"
    elif str(payee).startswith('EBAY'):
        return "eBay"
    elif str(payee).startswith('PAYPAL'):
        return "PayPal"
    elif str(payee).startswith('QANTAS'):
        return "Qantus"
    elif str(payee).startswith('ALASKA'):
        return "Alaska Airlines"
    elif str(payee).startswith('DELTA'):
        return "Delta Airlines"
    elif str(payee).startswith('UNITED'):
        return "United Airlines"
    elif str(payee).startswith('AMERICAN'):
        return "American Airlines"
    elif str(payee).startswith('SOUTHWEST'):
        return "Southwest Airlines"
    elif str(payee).startswith('UBER'):
        return "Uber"
    elif str(payee).startswith('LYFT'):
        return "Lyft"
    elif str(payee).startswith('DOORDASH'):
        return "DoorDash"
    elif str(payee).startswith('POSTMATE'):
        return "Postmate"
    elif str(payee).startswith('APPLE'):
        return "Apple"
    elif str(payee).startswith('CHEVRON'):
        return "Chevron"
    elif str(payee).startswith('SHELL'):
        return "Shell"
    elif str(payee).startswith('CVS'):
        return "CVS"
    elif str(payee).startswith('WALGREENS'):
        return "Walgreens"
    elif str(payee).startswith('WENDYS'):
        return "Wendy's"
    elif str(payee).startswith('MCDONALDS'):
        return "McDonald's"
    elif str(payee).startswith('BURGER KING'):
        return "Burger King"
    elif str(payee).startswith('THE UPS STORE'):
        return "UPS Store"

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


def create_payee_by_month_pivot(csv_folder='Csv', output_folder='Reports'):
    """
    Create a pivot table with payees as rows and months as columns.
    Shows total spending at each payee per month with totals.
    
    Args:
        csv_folder: Path to folder containing CSV files
        output_folder: Folder to save the report
        
    Returns:
        DataFrame with the pivot table
    """
    print("\n" + "=" * 80)
    print("CREATING PAYEE BY MONTH PIVOT TABLE")
    print("=" * 80)
    
    # Read all CSV files and normalize them
    all_data = read_all_csv_files_normalized(csv_folder)
    
    if all_data.empty:
        print("No data found!")
        return None
    
    print(f"\nTotal transactions loaded: {len(all_data)}")
    
    # Filter to only expenses (negative amounts)
    expenses = all_data[all_data['Amount'] < 0].copy()
    print(f"Total expense transactions: {len(expenses)}")
    
    # Clean merchant names
    expenses['Merchant'] = expenses['Payee'].apply(clean_merchant_name)
    
    # Extract month from date
    expenses['Month'] = expenses['Date'].dt.month
    expenses['Year'] = expenses['Date'].dt.year
    
    # Create pivot table: rows=Merchant, columns=Month, values=sum of Amount
    pivot = expenses.pivot_table(
        values='Amount',
        index='Merchant',
        columns='Month',
        aggfunc='sum',
        fill_value=0
    )
    
    # Rename month columns to month names
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
        5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
        9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    
    # Ensure all months are present (1-12), filling missing with 0
    for month in range(1, 13):
        if month not in pivot.columns:
            pivot[month] = 0.0
    
    # Reorder columns to be Jan-Dec
    pivot = pivot[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    pivot.columns = [month_names[m] for m in pivot.columns]
    
    # Add Total column (row totals - yearly total per payee)
    pivot['Total'] = pivot.sum(axis=1)
    
    # Sort by Total (most spending at top - since values are negative, sort ascending)
    pivot = pivot.sort_values('Total', ascending=True)
    
    # Add totals row at bottom
    totals_row = pivot.sum(axis=0)
    totals_row.name = 'TOTAL'
    pivot = pd.concat([pivot, pd.DataFrame([totals_row])])
    
    # Convert negative values to positive for display (optional - remove if you want to keep negatives)
    pivot_display = pivot.abs()
    
    # Re-sort by Total descending (most spent first) after converting to positive
    # Separate the TOTAL row, sort the rest, then add TOTAL back at bottom
    total_row = pivot_display.loc[['TOTAL']]
    data_rows = pivot_display.drop('TOTAL')
    data_rows = data_rows.sort_values('Total', ascending=False)
    pivot_display = pd.concat([data_rows, total_row])
    
    # Save to CSV
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    csv_file = output_path / 'payee_by_month_pivot.csv'
    pivot_display.to_csv(csv_file, encoding='utf-8-sig')
    print(f"\nPivot table saved to {csv_file}")
    
    # Also save as formatted text report
    txt_file = output_path / 'payee_by_month_pivot.txt'
    with open(txt_file, 'w') as f:
        f.write("PAYEE BY MONTH EXPENSE SUMMARY\n")
        f.write("=" * 150 + "\n\n")
        f.write(pivot_display.to_string())
        f.write("\n\n" + "=" * 150 + "\n")
        f.write(f"\nTotal unique payees: {len(pivot) - 1}\n")
        f.write(f"Total yearly expenses: ${pivot_display.loc['TOTAL', 'Total']:,.2f}\n")
    print(f"Text report saved to {txt_file}")
    
    # Save as JSON
    json_file = output_path / 'payee_by_month_pivot.json'
    pivot_display.to_json(json_file, orient='index', indent=2)
    print(f"JSON report saved to {json_file}")
    
    return pivot_display


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


def create_individual_monthly_charts(data_by_month, output_folder='Reports', top_n=15):
    """
    Create individual pie charts for each month showing top merchants.
    
    Args:
        data_by_month: Dictionary of DataFrames by month
        top_n: Number of top merchants to show individually
        output_folder: Folder to save charts
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    for month_name, df in data_by_month.items():
        merchant_totals = analyze_monthly_expenses(df, month_name)
        
        if not merchant_totals:
            continue
        
        # Get top N merchants and group the rest as "Other"
        sorted_merchants = sorted(merchant_totals.items(), key=lambda x: x[1])[:top_n]
        top_merchants = dict(sorted_merchants)
        
        # Calculate "Other" category if there are more merchants
        if len(merchant_totals) > top_n:
            other_total = sum(amount for merchant, amount in merchant_totals.items() 
                            if merchant not in top_merchants)
            top_merchants['Other'] = other_total
        
        # Convert to positive values for visualization
        merchants = list(top_merchants.keys())
        amounts = [-x for x in top_merchants.values()]
        
        # Create figure with two subplots: pie chart and bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Pie chart
        colors = plt.cm.Set3(range(len(merchants)))
        wedges, texts, autotexts = ax1.pie(amounts, labels=merchants, autopct='%1.1f%%',
                                           startangle=90, colors=colors, pctdistance=0.85)
        ax1.set_title(f'{month_name} - Expense Distribution', fontsize=14, fontweight='bold', pad=20)
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        # Bar chart
        bars = ax2.barh(range(len(merchants)), amounts, color=colors, 
                        edgecolor='gray', linewidth=1)
        ax2.set_yticks(range(len(merchants)))
        ax2.set_yticklabels(merchants, fontsize=9)
        ax2.set_xlabel('Amount Spent ($)', fontsize=11, fontweight='bold')
        ax2.set_title(f'{month_name} - Top Merchants', fontsize=14, fontweight='bold', pad=20)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, amounts)):
            ax2.text(value + max(amounts)*0.01, bar.get_y() + bar.get_height()/2,
                    f'${value:,.2f}', va='center', fontsize=9, fontweight='bold')
        
        # Add total at the top
        total = sum(amounts)
        fig.suptitle(f'Total Expenses: ${total:,.2f}', fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save chart
        chart_file = output_path / f'{month_name}_chart.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Chart saved for {month_name}")


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
    
    # Check if there's any data to plot
    if not sorted_months:
        print("Warning: No monthly data found matching expected month format. Skipping monthly expense chart.")
        return
    
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
    
    # Add average line (only if there's data)
    if len(sorted_totals_positive) > 0:
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
    # Check if yearly_totals is valid and not empty
    if not yearly_totals or not isinstance(yearly_totals, dict):
        print("Warning: No yearly totals data available. Skipping top merchants chart.")
        return
    
    if len(yearly_totals) == 0:
        print("Warning: Yearly totals dictionary is empty. Skipping top merchants chart.")
        return
    
    # Get top N merchants by spending
    sorted_merchants = sorted(yearly_totals.items(), key=lambda x: x[1])[:top_n]
    
    if not sorted_merchants:
        print("Warning: No merchants found in yearly totals. Skipping top merchants chart.")
        return
    
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

    csv_folder = 'Csv Chase'
    reports_folder = 'Reports Chase'
    
    # Read all CSV files
    data_by_month = read_csv_files(csv_folder)
    
    if not data_by_month:
        print(f"No CSV files found in the '{csv_folder}' folder!")
        return
    
    print(f"\nLoaded {len(data_by_month)} months of data")
    
    # Calculate monthly totals
    print("\nCalculating monthly totals...")
    monthly_totals = calculate_monthly_totals(data_by_month)
    
    # Save individual monthly reports
    print("\nGenerating monthly reports...")
    save_monthly_reports(data_by_month, reports_folder)
    
    # Aggregate yearly expenses
    print("\nAggregating yearly expenses...")
    yearly_totals = aggregate_yearly_expenses(data_by_month)
    
    # Save yearly report
    save_yearly_report(yearly_totals, reports_folder)
    
    # Create payee by month pivot table (includes all CSV formats)
    pivot_table = create_payee_by_month_pivot(csv_folder, reports_folder)
    
    # Create visualizations
    print("\nGenerating charts...")
    create_individual_monthly_charts(data_by_month, reports_folder)
    create_monthly_expense_chart(monthly_totals, reports_folder)
    create_top_merchants_chart(yearly_totals, output_folder=reports_folder)
    
    # Print summary
    print_summary(monthly_totals, yearly_totals)
    
    # Print pivot table preview
    if pivot_table is not None:
        print("\nPAYEE BY MONTH PIVOT TABLE (Top 20 payees):")
        print("-" * 150)
        print(pivot_table.head(20).to_string())
        print("-" * 150)
    
    print(f"Analysis complete! Check the '{reports_folder}' folder for detailed results.")


if __name__ == "__main__":
    main()
