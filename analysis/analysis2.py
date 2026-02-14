import pandas as pd
import numpy as np
import plotly.express as px
# تحميل البيانات
print("Loading merged dataset...")
df = pd.read_csv("E:/DS First Project/Dashboard/data/real_estate_tourism_merged.csv")

# تنظيف وتجهيز البيانات
df_clean = df.dropna(subset=[
    'area_name_en',
    'avg_meter_price',
    'tourism_activity',
    'transactions_count'
])

df_clean['year_month'] = pd.to_datetime(df_clean['year_month'])
df_clean['year'] = df_clean['year_month'].dt.year
df_clean['month'] = df_clean['year_month'].dt.month

print(f"Records: {len(df_clean)} | Areas: {df_clean['area_name_en'].nunique()}")

# حساب مؤشر الاستثمار المركب
investment_scores = []

for area in df_clean['area_name_en'].unique():
    area_data = df_clean[df_clean['area_name_en'] == area]

    if len(area_data) < 12:
        continue

    # نمو السياحة
    tourism_growth = 0
    if len(area_data) >= 24:
        recent = area_data[area_data['year'] >= 2022]['tourism_activity'].mean()
        old = area_data[area_data['year'] < 2022]['tourism_activity'].mean()
        if old > 0:
            tourism_growth = ((recent - old) / old) * 100

    # استقرار الأسعار
    price_stability = 0
    price_std = area_data['avg_meter_price'].std()
    price_mean = area_data['avg_meter_price'].mean()
    if price_mean > 0:
        price_stability = 100 * (1 - price_std / price_mean)

    # السيولة
    liquidity = area_data['transactions_count'].sum() / len(area_data)

    # مستوى السياحة الحالي
    current_tourism = area_data['tourism_activity'].mean()
    tourism_percentile = (
        (current_tourism - df_clean['tourism_activity'].min()) /
        (df_clean['tourism_activity'].max() - df_clean['tourism_activity'].min())
    ) * 100

    # جاذبية السعر
    price_attractiveness = 0
    overall_mean_price = df_clean['avg_meter_price'].mean()
    area_mean_price = area_data['avg_meter_price'].mean()
    if overall_mean_price > 0:
        price_attractiveness = max(0, 100 * (1 - area_mean_price / overall_mean_price))

    # المؤشر المركب
    composite_score = (
        min(max(tourism_growth, -50), 100) * 0.30 +
        max(price_stability, 0) * 0.25 +
        min(liquidity * 10, 100) * 0.20 +
        tourism_percentile * 0.15 +
        price_attractiveness * 0.10
    )

    investment_scores.append({
        'Area': area,
        'Investment Score': round(composite_score, 2),
        'Tourism Growth %': round(tourism_growth, 2),
        'Price Stability %': round(price_stability, 2),
        'Monthly Liquidity': round(liquidity, 2),
        'Tourism Level': round(current_tourism, 2),
        'Avg Meter Price': round(area_mean_price, 2),
        'Months': len(area_data),
        'Transactions': int(area_data['transactions_count'].sum())
    })

scores_df = pd.DataFrame(investment_scores)

if scores_df.empty:
    print("No valid areas for scoring")
    exit()

scores_df = scores_df.sort_values('Investment Score', ascending=False)
print("Investment scoring completed")

# تصنيف الاستثمار
def classify(score):
    if score >= 70:
        return "Excellent"
    elif score >= 60:
        return "Very Good"
    elif score >= 50:
        return "Good"
    elif score >= 40:
        return "Average"
    else:
        return "Weak"

scores_df['Rating'] = scores_df['Investment Score'].apply(classify)

# تحليل فئات السعر
price_bins = [0, 5000, 10000, 20000, 50000, float('inf')]
price_labels = ['Low', 'Medium', 'High', 'Very High', 'Luxury']
scores_df['Price Segment'] = pd.cut(scores_df['Avg Meter Price'],
                                    bins=price_bins,
                                    labels=price_labels)

print("Price segmentation completed")

# تحديد الفرص الخاصة
emerging_areas = scores_df[
    (scores_df['Tourism Growth %'] > 20) &
    (scores_df['Avg Meter Price'] < df_clean['avg_meter_price'].mean())
]

stable_areas = scores_df[
    (scores_df['Price Stability %'] > 80) &
    (scores_df['Monthly Liquidity'] > 3)
]

print(f"Emerging areas: {len(emerging_areas)} | Stable areas: {len(stable_areas)}")


#Drawing the charts
#Chart 1
top_10 = scores_df.head(10)

chart1 = px.bar(top_10, x = "Investment Score", y = "Area", orientation="h", title = "Top 10 Investment Areas")

chart1.update_layout(title_x = 0.5, yaxis = dict(autorange = "reversed"), plot_bgcolor = "white")

#Converting the chart to an html file (interactive)
chart1.write_html("charts/chart-2.1.html", include_plotlyjs = "cdn")

#Chart2
rating_counts = scores_df['Rating'].value_counts().reset_index()
rating_counts.columns = ["Rating", "Count"]

chart2 = px.pie(rating_counts, names = "Rating", values = "Count", title = "Investment Rating Distribution")

chart2.update_layout(title_x = 0.5)
chart2.update_traces(textinfo = "percent+label")

chart2.write_html("charts/chart-2.2.html", include_plotlyjs = "cdn")

#Chart3
chart3 = px.scatter(scores_df, x = "Avg Meter Price", y = "Investment Score", color = "Tourism Growth %", color_continuous_scale = "RdYlGn", title = "Price vs Investment Score (Tourism Growth Colored)", opacity = 0.7)

chart3.update_layout(title_x = 0.5, plot_bgcolor = "white")

chart3.write_html("charts/chart-2.3.html", include_plotlyjs = "cdn")