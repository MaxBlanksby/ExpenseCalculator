# Expense Analysis Tool

This tool analyzes CSV files containing bank transaction data and generates comprehensive expense reports.

## Features

- ✅ Reads all CSV files from a specified folder
- ✅ Tallies expenses by merchant/store for each month
- ✅ Generates individual monthly expense reports
- ✅ Creates yearly aggregated expense report
- ✅ Generates bar chart showing total monthly expenses
- ✅ Creates top merchants spending chart
- ✅ Filters only expenses (ignores income/deposits)

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place all your CSV files in the `Csv` folder
2. Run the analysis:
```bash
python analyze_expenses.py
```

3. View the results in the `Reports` folder:
   - `monthly_expenses.txt` - Individual monthly reports
   - `yearly_expenses.txt` - Full year aggregated report
   - `monthly_expenses_chart.png` - Bar chart of monthly totals
   - `top_merchants_chart.png` - Top merchants by spending

## CSV File Format

The script expects CSV files with the following columns:
- Posted Date
- Reference Number
- Payee (merchant/store name)
- Address
- Amount (negative for expenses, positive for income)

## Customization

You can modify the `analyze_expenses.py` script to:
- Change the number of top merchants displayed (default: 15)
- Adjust merchant name cleaning rules
- Modify chart styles and colors
- Change output folder locations

## Output Structure

```
ExpensesCalc/
├── Csv/                          # Your CSV files here
├── Reports/                      # Generated reports
│   ├── January2025_expenses.txt
│   ├── January2025_expenses.json
│   ├── ... (one for each month)
│   ├── yearly_expenses.txt
│   ├── yearly_expenses.json
│   ├── monthly_expenses_chart.png
│   └── top_merchants_chart.png
├── analyze_expenses.py           # Main script
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Notes

- Only negative amounts (expenses) are included in the analysis
- Positive amounts (income, refunds) are automatically excluded
- Merchant names are automatically cleaned for better grouping
- Charts are saved as high-resolution PNG files (300 DPI)
