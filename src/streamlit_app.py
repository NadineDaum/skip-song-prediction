import streamlit as st
import pandas as pd
import plotly.express as px
import duckdb

st.set_page_config(page_title="Skip Behavior Dashboard", layout="wide")

# Load data
df = pd.read_csv("data/synthetic_sessions.csv")

# CLEAN CATEGORIES BEFORE USING THEM IN FILTERS
df["country"]       = df["country"].astype(str)
df["platform"]      = df["platform"].astype(str)
df["subscription"]  = df["subscription"].astype(str)
df["genre"]         = df["genre"].astype(str)
df["age_group"]     = df["age_group"].astype(str)
df["time_of_day"]   = df["time_of_day"].astype(str)
df["day_type"]      = df["day_type"].astype(str)
df["position"]      = df["position"].astype(int)

# Additional safeguard
cat_cols = [
    "country", "platform", "subscription", "genre",
    "age_group", "time_of_day", "day_type"
]
for col in cat_cols:
    df[col] = df[col].astype(str)


st.sidebar.title("Filters")

# --- Sidebar Filters ---
country_filter   = st.sidebar.multiselect("Country", sorted(df["country"].unique()))
platform_filter  = st.sidebar.multiselect("Platform", sorted(df["platform"].unique()))
age_filter       = st.sidebar.multiselect("Age Group", sorted(df["age_group"].unique()))
sub_filter       = st.sidebar.multiselect("Subscription", sorted(df["subscription"].unique()))
genre_filter     = st.sidebar.multiselect("Genre", sorted(df["genre"].unique()))
time_filter      = st.sidebar.multiselect("Time of Day", sorted(df["time_of_day"].unique()))
daytype_filter   = st.sidebar.multiselect("Day Type", sorted(df["day_type"].unique()))
pos_range        = st.sidebar.slider("Track Position", 0, 40, (0, 40))

# --- Apply Filters ---
df_filtered = df.copy()

if country_filter:
    df_filtered = df_filtered[df_filtered["country"].isin(country_filter)]
if platform_filter:
    df_filtered = df_filtered[df_filtered["platform"].isin(platform_filter)]
if age_filter:
    df_filtered = df_filtered[df_filtered["age_group"].isin(age_filter)]
if sub_filter:
    df_filtered = df_filtered[df_filtered["subscription"].isin(sub_filter)]
if genre_filter:
    df_filtered = df_filtered[df_filtered["genre"].isin(genre_filter)]
if time_filter:
    df_filtered = df_filtered[df_filtered["time_of_day"].isin(time_filter)]
if daytype_filter:
    df_filtered = df_filtered[df_filtered["day_type"].isin(daytype_filter)]

df_filtered = df_filtered[
    (df_filtered["position"] >= pos_range[0]) &
    (df_filtered["position"] <= pos_range[1])
]

# --- KPI ---
skip_rate = df_filtered["skip"].mean() * 100

st.markdown(f"""
# 🎧 Skip Behavior Dashboard  
### Overall Skip Rate: **{skip_rate:.2f}%**
""")

# --- Grid of charts ---
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# 1. Age group
age_data = df_filtered.groupby("age_group")["skip"].mean().reset_index()
fig_age = px.line(age_data, x="age_group", y="skip", markers=True, title="Skip Rate by Age Group")
col1.plotly_chart(fig_age, use_container_width=True)

# 2. Track position
pos_data = df_filtered.groupby("position")["skip"].mean().reset_index()
fig_pos = px.line(pos_data, x="position", y="skip", markers=True, title="Skip Rate by Track Position")
col2.plotly_chart(fig_pos, use_container_width=True)

# 3. Time of day
tod_data = df_filtered.groupby("time_of_day")["skip"].mean().reset_index()
fig_tod = px.bar(tod_data, x="time_of_day", y="skip", title="Skip Rate by Time of Day")
col3.plotly_chart(fig_tod, use_container_width=True)

# 4. Genre
genre_data = df_filtered.groupby("genre")["skip"].mean().reset_index()
fig_genre = px.bar(genre_data, x="genre", y="skip", title="Skip Rate by Genre")
col4.plotly_chart(fig_genre, use_container_width=True)
