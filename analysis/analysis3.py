import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# تحميل البيانات
print("Loading data...")
df = pd.read_csv("E:/DS First Project/Dashboard/data/real_estate_tourism_merged.csv")

# تنظيف وتجهيز البيانات الزمنية
df_clean = df.dropna(subset=[
    'year_month',
    'avg_meter_price',
    'tourism_activity',
    'transactions_count'
])

df_clean['year_month'] = pd.to_datetime(df_clean['year_month'])
df_clean['year'] = df_clean['year_month'].dt.year
df_clean['month'] = df_clean['year_month'].dt.month
df_clean['quarter'] = df_clean['year_month'].dt.quarter

print(f"Records: {len(df_clean)} | Years: {df_clean['year'].min()}–{df_clean['year'].max()}")

# التحليل الشهري المجمع
monthly_stats = df_clean.groupby(['year', 'month']).agg({
    'avg_meter_price': 'mean',
    'tourism_activity': 'mean',
    'transactions_count': 'sum',
    'area_name_en': 'nunique'
}).reset_index()

# أسماء الشهور
month_names = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
    5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
    9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}

monthly_stats['month_name'] = monthly_stats['month'].map(month_names)

# حساب الأنماط الشهرية عبر السنوات
monthly_patterns = monthly_stats.groupby('month').agg({
    'avg_meter_price': ['mean', 'std', 'min', 'max'],
    'tourism_activity': 'mean',
    'transactions_count': 'mean',
    'area_name_en': 'mean'
}).round(2)

monthly_patterns.columns = ['_'.join(c) for c in monthly_patterns.columns]
monthly_patterns = monthly_patterns.reset_index()
monthly_patterns['month_name'] = monthly_patterns['month'].map(month_names)

print("Monthly patterns ready")

# حساب نقاط الشراء
def calculate_month_score(row):
    score = 0

    price_percentile = (
        row['avg_meter_price_mean'] - monthly_patterns['avg_meter_price_mean'].min()
    ) / (
        monthly_patterns['avg_meter_price_mean'].max() -
        monthly_patterns['avg_meter_price_mean'].min()
    )
    score += price_percentile * 40

    volatility = (
        row['avg_meter_price_std'] / row['avg_meter_price_mean']
        if row['avg_meter_price_mean'] > 0 else 1
    )
    score += min(volatility * 100, 30)

    transactions_percentile = (
        row['transactions_count_mean'] - monthly_patterns['transactions_count_mean'].min()
    ) / (
        monthly_patterns['transactions_count_mean'].max() -
        monthly_patterns['transactions_count_mean'].min()
    )
    score += transactions_percentile * 20

    tourism_percentile = (
        row['tourism_activity_mean'] - monthly_patterns['tourism_activity_mean'].min()
    ) / (
        monthly_patterns['tourism_activity_mean'].max() -
        monthly_patterns['tourism_activity_mean'].min()
    )
    score += tourism_percentile * 10

    return round(score, 2)

monthly_patterns['buy_score'] = monthly_patterns.apply(calculate_month_score, axis=1)
monthly_patterns = monthly_patterns.sort_values('buy_score')

print("Month ranking calculated")

# مقارنة الشتاء والصيف
winter = df_clean[df_clean['month'].isin([12, 1, 2])]
summer = df_clean[df_clean['month'].isin([6, 7, 8])]

if not winter.empty and not summer.empty:
    winter_price = winter['avg_meter_price'].mean()
    summer_price = summer['avg_meter_price'].mean()

    better_season = "Summer" if summer_price < winter_price else "Winter"
    print(f"Seasonal comparison done | Better buying season: {better_season}")

# تحديد التوقيت حسب نوع العقار
property_timing = {}

for prop in df_clean['property_type_en'].dropna().unique():
    prop_data = df_clean[df_clean['property_type_en'] == prop]

    if len(prop_data) < 100:
        continue

    prop_monthly = prop_data.groupby('month').agg({
        'avg_meter_price': 'mean',
        'transactions_count': 'sum'
    }).reset_index()

    cheapest = prop_monthly.loc[prop_monthly['avg_meter_price'].idxmin()]
    busiest = prop_monthly.loc[prop_monthly['transactions_count'].idxmax()]

    property_timing[prop] = {
        'best_price_month': month_names[cheapest['month']],
        'best_price': cheapest['avg_meter_price'],
        'highest_activity_month': month_names[busiest['month']],
        'activity': busiest['transactions_count'],
        'saving_pct': (
            (prop_monthly['avg_meter_price'].max() - cheapest['avg_meter_price']) /
            cheapest['avg_meter_price']
        ) * 100
    }

print(f"Property timing calculated for {len(property_timing)} types")

# تحميل البيانات
print("Loading data...")
df = pd.read_csv(
    "E:/DS First Project/Dashboard/data/real_estate_tourism_merged.csv"
)

# تنظيف القيم الأساسية
df = df.dropna(subset=[
    'area_name_en',
    'avg_meter_price',
    'tourism_activity',
    'transactions_count',
    'year_month'
])

df['year_month'] = pd.to_datetime(df['year_month'])
df = df.sort_values(['area_name_en', 'year_month'])

print(f"Records loaded: {len(df)}")
print(f"Areas detected: {df['area_name_en'].nunique()}")

# تنعيم الأسعار والنشاط السياحي
df['price_smooth'] = (
    df.groupby('area_name_en')['avg_meter_price']
    .transform(lambda x: x.rolling(6, min_periods=3).mean())
)

df['tourism_smooth'] = (
    df.groupby('area_name_en')['tourism_activity']
    .transform(lambda x: x.rolling(6, min_periods=3).mean())
)

# إزاحة السياحة زمنياً (تأثير متأخر)
df['tourism_lag_3'] = (
    df.groupby('area_name_en')['tourism_smooth']
    .shift(3)
)

# تحليل المخاطر المركبة
risk_rows = []

for area in df['area_name_en'].unique():
    area_df = df[df['area_name_en'] == area].dropna()

    if len(area_df) < 18:
        continue

    risk_score = 0
    notes = []

    price_mean = area_df['price_smooth'].mean()
    price_vol = area_df['price_smooth'].std() / price_mean

    if price_vol > 0.4:
        risk_score += 25
        notes.append("High price volatility")

    corr_lag = area_df['price_smooth'].corr(area_df['tourism_lag_3'])
    if abs(corr_lag) > 0.6:
        risk_score += 25
        notes.append("Lagged tourism sensitivity")

    avg_tx = area_df['transactions_count'].mean()
    if avg_tx < 2:
        risk_score += 20
        notes.append("Low liquidity")

    market_price = df['price_smooth'].mean()
    if price_mean > market_price * 1.4:
        risk_score += 15
        notes.append("Above market pricing")

    risk_rows.append({
        'area': area,
        'risk_score': risk_score,
        'price_volatility': round(price_vol, 3),
        'tourism_corr_lagged': round(corr_lag, 3),
        'avg_price': round(price_mean, 2),
        'avg_transactions': round(avg_tx, 2),
        'notes': " | ".join(notes)
    })

risk_df = pd.DataFrame(risk_rows).sort_values('risk_score', ascending=False)
print(f"High risk areas detected: {len(risk_df[risk_df['risk_score'] >= 50])}")

# تحليل الاعتماد على السياحة المتأخرة
dependency_rows = []

for area in df['area_name_en'].unique():
    area_df = df[df['area_name_en'] == area].dropna()

    if len(area_df) < 18:
        continue

    corr = area_df['price_smooth'].corr(area_df['tourism_lag_3'])

    dependency_rows.append({
        'area': area,
        'tourism_dependency_lagged': round(corr, 3),
        'avg_price': round(area_df['price_smooth'].mean(), 2)
    })

dependency_df = pd.DataFrame(dependency_rows)
print("Dependency analysis completed")

# تحليل استقرار الأسعار
stability_rows = []

for area in df['area_name_en'].unique():
    area_df = df[df['area_name_en'] == area].dropna()

    if len(area_df) < 24:
        continue

    price_mean = area_df['price_smooth'].mean()
    price_std = area_df['price_smooth'].std()
    price_cv = (price_std / price_mean) * 100 if price_mean > 0 else 0

    if price_cv < 15:
        stability = "Very Stable"
    elif price_cv < 25:
        stability = "Stable"
    elif price_cv < 40:
        stability = "Moderate"
    else:
        stability = "Volatile"

    stability_rows.append({
        'area': area,
        'price_volatility_%': round(price_cv, 2),
        'stability_class': stability,
        'avg_price': round(price_mean, 2),
        'transactions': int(area_df['transactions_count'].sum())
    })

stability_df = pd.DataFrame(stability_rows).sort_values('price_volatility_%')
print(f"Stability analysis completed: {len(stability_df)} areas")

#Drawing the charts
#Chart 1
monthly_pivot = monthly_stats.pivot(index = 'month', columns = 'year', values = 'avg_meter_price')

monthly_pivot.index = monthly_pivot.index.map(month_names)
monthly_pivot = monthly_pivot.reset_index()

chart1 = px.line(monthly_pivot, x = 'month', y = monthly_pivot.columns[1:], title = "Monthly Price Trends")

chart1.update_layout(title_x = 0.5, plot_bgcolor = "white", xaxis_title = "Month", yaxis_title = "Average Meter Price")

chart1.write_html("charts/chart-3.1.html", include_plotlyjs = "cdn")

#Chart 2
sorted_months = monthly_patterns.sort_values('buy_score')

chart2 = px.bar(sorted_months, x = "buy_score", y = "month_name", orientation = "h", title = "Best Months to Buy")

chart2.update_layout(title_x = 0.5, yaxis_autorange = "reversed", plot_bgcolor = "white", xaxis_title = "Buy Score", yaxis_title = "Month Name")

chart2.write_html("charts/chart-3.2.html", include_plotlyjs = "cdn")

#Chart 3
season_df = pd.DataFrame({"Season": ["Winter", "Summer"],"Average Price": [winter_price, summer_price]})

chart3 = px.bar(season_df, x = "Season", y = "Average Price", title = "Seasonal Price Comparison")

chart3.update_layout(title_x = 0.5, plot_bgcolor="white")

chart3.write_html("charts/chart-3.3.html", include_plotlyjs = "cdn")

#Chart 4
top_risk = risk_df.head(15)

chart4 = px.bar(top_risk, x = "risk_score", y = "area", orientation = "h", title = "Top High Risk Areas", color_discrete_sequence = ["#e74c3c"])

chart4.update_layout(title_x = 0.5, yaxis_autorange = "reversed", plot_bgcolor = "white", xaxis_title = "Risk Score", yaxis_title = "Area")

chart4.write_html("charts/chart-3.4.html", include_plotlyjs = "cdn")

#Chart 5
chart5 = px.scatter(dependency_df, x = "tourism_dependency_lagged", y = "avg_price", title = "Lagged Tourism Sensitivity vs Avg Price", opacity = 0.7, color_discrete_sequence = ["#3498db"])

chart5.add_vline(x = 0, line_dash = "dash", line_color = "red", opacity = 0.6)

chart5.update_layout(title_x = 0.5, plot_bgcolor = "white", xaxis_title = "Lagged Tourism Dependency", yaxis_title = "Average Price")

chart5.write_html("charts/chart-3.5.html", include_plotlyjs = "cdn")

#Chart 6
stability_counts = stability_df['stability_class'].value_counts().reset_index()
stability_counts.columns = ["Stability Class", "Count"]

chart6 = px.pie(stability_counts, names = "Stability Class", values = "Count", title = "Price Stability Distribution", color_discrete_sequence = ["#27ae60", "#3498db", "#f39c12", "#e74c3c"])

chart6.update_layout(title_x = 0.5)
chart6.update_traces(textinfo = "percent+label")

chart6.write_html("charts/chart-3.6.html", include_plotlyjs="cdn")