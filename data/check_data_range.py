import pandas as pd

df = pd.read_csv('monthly_data_8cat_no_covid.csv')
df['month'] = pd.to_datetime(df['month'])

print('Production Training Data - Detailed Breakdown')
print('='*60)
print(f'First month: {df.month.min().strftime("%Y-%m")}')
print(f'Last month: {df.month.max().strftime("%Y-%m")}')
print(f'Total months: {len(df)}')

print('\n' + '='*60)
print('Months by Year:')
print('='*60)

for year in sorted(df.month.dt.year.unique()):
    year_data = df[df.month.dt.year == year]
    months = year_data.month.dt.strftime('%Y-%m').tolist()
    print(f'\n{year} ({len(year_data)} months):')
    print(', '.join(months))

# Check for gaps
print('\n' + '='*60)
print('EXCLUDED PERIODS (gaps in data):')
print('='*60)

all_months = df.month.sort_values().tolist()
for i in range(len(all_months) - 1):
    current = all_months[i]
    next_month = all_months[i + 1]
    expected_next = current + pd.DateOffset(months=1)
    
    if next_month != expected_next:
        gap_start = current + pd.DateOffset(months=1)
        gap_end = next_month - pd.DateOffset(months=1)
        print(f'\nGap: {gap_start.strftime("%Y-%m")} to {gap_end.strftime("%Y-%m")}')
        
        # Calculate number of months in gap
        months_gap = (next_month.year - gap_start.year) * 12 + (next_month.month - gap_start.month)
        print(f'  → {months_gap} months excluded')
