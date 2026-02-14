import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px

# تحميل البيانات
print("Loading merged dataset...")
df = pd.read_csv("E:/DS First Project/Dashboard/data/real_estate_tourism_merged.csv")

# تنظيف أولي
df_clean = df.dropna(subset=['tourism_activity', 'avg_meter_price',
                             'property_type_en', 'area_name_en'])
print(f"Records used for analysis: {len(df_clean)}")

# 1 العلاقة العامة بين السياحة والسعر
correlation = df_clean['tourism_activity'].corr(df_clean['avg_meter_price'])
p_value = stats.pearsonr(df_clean['tourism_activity'],
                          df_clean['avg_meter_price'])[1]

print(f"Overall correlation: {correlation:.4f}")
print(f"P-value: {p_value:.6f}")

# 2 العلاقة حسب نوع العقار
property_results = []
for prop in df_clean['property_type_en'].unique():
    subset = df_clean[df_clean['property_type_en'] == prop]
    if len(subset) > 50:
        property_results.append({
            'Property Type': prop,
            'Correlation': subset['tourism_activity'].corr(subset['avg_meter_price']),
            'Transactions': len(subset),
            'Avg Tourism': subset['tourism_activity'].mean(),
            'Avg Price': subset['avg_meter_price'].mean()
        })

property_df = pd.DataFrame(property_results)
if not property_df.empty:
    property_df = property_df.sort_values('Correlation', ascending=False)
    print("Property-type analysis completed")

# 3 تحليل المناطق
area_results = []
for area in df_clean['area_name_en'].unique():
    area_data = df_clean[df_clean['area_name_en'] == area]
    if len(area_data) > 30:
        area_results.append({
            'Area': area,
            'Correlation': area_data['tourism_activity'].corr(area_data['avg_meter_price']),
            'Observations': len(area_data),
            'Avg Tourism': area_data['tourism_activity'].mean(),
            'Avg Price': area_data['avg_meter_price'].mean()
        })

area_df = pd.DataFrame(area_results)

def classify(c):
    if c > 0.5:
        return "Very Strong"
    elif c > 0.3:
        return "Moderate"
    elif c > 0.1:
        return "Weak"
    elif c > -0.1:
        return "No Clear Impact"
    else:
        return "Negative"

if not area_df.empty:
    area_df['Impact Class'] = area_df['Correlation'].apply(classify)
    top_10 = area_df.sort_values('Correlation', ascending=False).head(10)
    print("Top impacted areas identified")

#Drawing the charts
#Chart 1

chart1 = px.scatter(df_clean, x = "tourism_activity", y = "avg_meter_price", opacity = 0.3, title = f"Overall Relationship (corr={correlation:.3f})")

chart1.update_traces(marker_size = 6)

chart1.update_layout(title_x = 0.5, plot_bgcolor = "white", xaxis_title = "Tourism Activity", yaxis_title = "Average Meter Price")

chart1.write_html("charts/chart-1.1.html", include_plotlyjs = "cdn")

#Chart 2
if not property_df.empty:

    chart2 = px.bar(property_df, x = "Property Type", y = "Correlation", title = "Correlation by Property Type")

    chart2.add_hline(y = 0, line_dash = "dash")

    chart2.update_layout(title_x = 0.5, plot_bgcolor = "white")

    chart2.write_html("charts/chart-1.2.html", include_plotlyjs = "cdn")

#Chart 3
if not area_df.empty:

    chart3 = px.bar(top_10, x = "Correlation", y = "Area", orientation = "h", title = "Top 10 Areas by Tourism Impact")

    chart3.add_vline(x = 0, line_dash = "dash")

    chart3.update_layout(title_x = 0.5, yaxis_autorange = "reversed", plot_bgcolor = "white")

    chart3.write_html("charts/chart-1.3.html", include_plotlyjs = "cdn")

#Chart 4
impact_counts = area_df['Impact Class'].value_counts().reset_index()
impact_counts.columns = ["Impact Class", "Count"]

chart4 = px.pie(impact_counts, names = "Impact Class", values = "Count", title = "Impact Distribution")

chart4.update_layout(title_x = 0.5)
chart4.update_traces(textinfo = "percent+label")

chart4.write_html("charts/chart-1.4.html", include_plotlyjs = "cdn")