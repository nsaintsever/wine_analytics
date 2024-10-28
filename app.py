import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from currency_converter import CurrencyConverter

st.set_page_config(layout="wide")

df = pd.read_csv("lacoste_sub_df.csv", sep=',', quotechar='"')

df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

df['quantite_totale'] = df['per'] * df['quantite']

df['millesime'] = df['millesime'].fillna(0).astype(int)

millesimes_sorted = sorted(df['millesime'].unique(), reverse=True)

currency_converter = CurrencyConverter()

def get_conversion_rate(date, from_currency="GBP", to_currency="EUR"):
    try:
        rate = currency_converter.convert(1, from_currency, to_currency, date=date)
    except:
        rate = None
    return rate

df['conversion_rate'] = df['date'].apply(lambda x: get_conversion_rate(x))

df['conversion_rate'] = df['conversion_rate'].fillna(method='ffill')

df['prix_eur'] = df['prix_unitaire'] * df['conversion_rate']

def get_longest_vin(vin_clean_group):
    return max(vin_clean_group, key=len)

vin_clean_options = df.groupby('vin_clean')['vin'].apply(get_longest_vin).to_dict()
vin_options = {v: k for k, v in vin_clean_options.items()}
last_date_in_data = df['date'].max()

def permissive_padding(df_vin, column_name):
    full_dates = pd.date_range(start=df_vin['date'].min(), end=df_vin['date'].max())
    df_vin_padded = pd.DataFrame()
    for negociant in df_vin['negociant'].unique():
        df_negociant = df_vin[df_vin['negociant'] == negociant]
        df_negociant = df_negociant.drop_duplicates(subset=['date'])
        df_negociant = df_negociant.set_index('date').reindex(full_dates, fill_value=None)
        df_negociant['negociant'] = negociant
        df_negociant[column_name] = df_negociant[column_name].fillna(method='ffill')
        df_vin_padded = pd.concat([df_vin_padded, df_negociant])
    return df_vin_padded.reset_index().rename(columns={'index': 'date'})

st.sidebar.title("Options de sélection")

vin_selected_display = st.sidebar.selectbox("Choisissez un vin", list(vin_options.keys()))
vin_selected = vin_options[vin_selected_display]

millesime_selected = st.sidebar.selectbox("Choisissez un millésime", millesimes_sorted)
format_options = df[(df['vin_clean'] == vin_selected) & (df['millesime'] == millesime_selected)]['format'].unique()
format_selected = st.sidebar.selectbox("Choisissez un format", format_options, index=0)

date_selected = st.sidebar.date_input("Choisissez une date", last_date_in_data.date(), min_value=df['date'].min(), max_value=last_date_in_data)

df_vin = df[(df['vin_clean'] == vin_selected) & 
            (df['millesime'] == millesime_selected) & 
            (df['format'] == format_selected)]

df_vin = permissive_padding(df_vin, 'prix_unitaire')
df_vin = permissive_padding(df_vin, 'quantite_totale')

df_vin['prix_unitaire_pondere'] = df_vin['prix_unitaire'] * df_vin['quantite_totale']

df_stats = df_vin.groupby('date').apply(lambda x: pd.Series({
    'total_quantity': x['quantite_totale'].sum(),
    'weighted_mean_prix_unitaire': (x['prix_unitaire_pondere'].sum()) / x['quantite_totale'].sum(),
    'prix_min': x['prix_unitaire'].min(),
    'prix_max': x['prix_unitaire'].max(),
    'prix_q1': x['prix_unitaire'].quantile(0.25),
    'prix_median': x['prix_unitaire'].median(),
    'prix_q3': x['prix_unitaire'].quantile(0.75),
    'conversion_rate': x['conversion_rate'].iloc[0]  # Assuming unique conversion rate per date
})).reset_index()


df_stats['weighted_mean_prix_eur'] = df_stats['weighted_mean_prix_unitaire'] * df_stats['conversion_rate']
df_stats['prix_eur_min'] = df_stats['prix_min'] * df_stats['conversion_rate']
df_stats['prix_eur_max'] = df_stats['prix_max'] * df_stats['conversion_rate']
df_stats['prix_eur_q1'] = df_stats['prix_q1'] * df_stats['conversion_rate']
df_stats['prix_eur_median'] = df_stats['prix_median'] * df_stats['conversion_rate']
df_stats['prix_eur_q3'] = df_stats['prix_q3'] * df_stats['conversion_rate']

df_stats = df_stats.fillna(method="ffill").fillna(method="bfill")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"Distribution des prix pour {vin_selected_display}")

    fig_price = go.Figure()

    fig_price.add_trace(go.Scatter(
        x=df_stats['date'],
        y=df_stats['weighted_mean_prix_unitaire'],
        mode='lines',
        name="Prix GBP",
        line=dict(color='maroon'),
        hovertemplate=(
            '<b>Prix Min:</b> %{customdata[0]:.2f}<br>'
            '<b>Prix Max:</b> %{customdata[1]:.2f}<br>'
            '<b>Q1:</b> %{customdata[2]:.2f}<br>'
            '<b>Moyenne Pondérée:</b> %{customdata[5]:.2f}<br>'
            '<b>Médiane:</b> %{customdata[3]:.2f}<br>'
            '<b>Q3:</b> %{customdata[4]:.2f}<br>'
        ),
        customdata=df_stats[['prix_min', 'prix_max', 'prix_q1', 'prix_median', 'prix_q3', 'weighted_mean_prix_unitaire']]
    ))

    fig_price.add_trace(go.Scatter(
        x=df_stats['date'],
        y=df_stats['weighted_mean_prix_eur'],
        mode='lines',
        name="Équivalent EUR",
        line=dict(color='goldenrod', dash='solid'),
        hovertemplate=(
            '<b>Prix Min (EUR):</b> %{customdata[0]:.2f}<br>'
            '<b>Prix Max (EUR):</b> %{customdata[1]:.2f}<br>'
            '<b>Q1 (EUR):</b> %{customdata[2]:.2f}<br>'
            '<b>Moyenne Pondérée (EUR):</b> %{customdata[5]:.2f}<br>'
            '<b>Médiane (EUR):</b> %{customdata[3]:.2f}<br>'
            '<b>Q3 (EUR):</b> %{customdata[4]:.2f}<br>'
        ),
        customdata=df_stats[['prix_eur_min', 'prix_eur_max', 'prix_eur_q1', 'prix_eur_median', 'prix_eur_q3', 'weighted_mean_prix_eur']]
    ))

    fig_price.update_layout(
        title=f"Distribution des prix pour {vin_selected_display}",
        xaxis_title="Date",
        yaxis_title="Prix Unitaire (GBP/EUR)",
        hovermode="x unified",
        yaxis=dict(range=[0, max(df_stats['prix_max'].max(), df_stats['prix_eur_max'].max()) + 10])
    )

    st.plotly_chart(fig_price)
    


st.markdown(
    """
    <style>
    div[data-testid="column"]:nth-child(1) {
        border-right: 1px solid rgba(255, 255, 255, 0.5);
        padding-right: 10px;
    }
    div[data-testid="column"]:nth-child(2) {
        padding-left: 10px;
    }
    .stDataFrame { 
        margin-left: 150px; 
        margin-right: auto;
        margin-top: 50px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

with col2:
    st.subheader(f"Ventilation des stocks pour {vin_selected_display} à la date du : {date_selected}")

    df_stock = df_vin[df_vin['date'] == pd.to_datetime(date_selected)].groupby('negociant').agg(
        {'quantite_totale': 'sum', 'prix_unitaire': 'mean'}
    ).reset_index()

    df_stock = df_stock[df_stock['quantite_totale'] > 0]

    df_stock = df_stock.sort_values(by='prix_unitaire', ascending=True)

    df_stock['quantite_totale'] = df_stock['quantite_totale'].astype(int)
    df_stock['prix_unitaire'] = df_stock['prix_unitaire'].round(2)

    df_stock.columns = ['Négociants', 'Stocks', 'Prix']

    total_volume = df_stock['Stocks'].sum()
    df_stock.loc[len(df_stock.index)] = ['Total', total_volume, '']  # Ajout d'une ligne pour les totaux

    styled_df = df_stock.style.hide(axis='index').format(
        {'Prix': lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x}
    ).set_table_styles(
        [{'selector': 'thead th', 'props': [('background-color', '#ADD8E6'), ('color', 'black'), ('font-weight', 'bold')]}]
    )

    st.write(styled_df)


st.subheader(f"Évolution des volumes totaux dans le temps pour {vin_selected_display}")

df_volume_time = df_vin.groupby(['date', 'negociant']).agg({'quantite_totale': 'sum'}).reset_index()

df_negociants = df_volume_time.groupby('date').apply(lambda x: '<br>'.join([f"{row['negociant']}: {row['quantite_totale']}" for _, row in x.iterrows()])).reset_index(name='ventilation_negociant')

df_volume_sum = df_volume_time.groupby('date').agg({'quantite_totale': 'sum'}).reset_index()

df_volume_sum = pd.merge(df_volume_sum, df_negociants, on='date')

fig_volumes = go.Figure()

fig_volumes.add_trace(go.Scatter(
    x=df_volume_sum['date'], 
    y=df_volume_sum['quantite_totale'], 
    line=dict(color='darkgoldenrod'),
    mode='lines+markers',
    name="Volume total",
    hovertemplate='%{y} bouteilles<br>%{text}',  
    text=df_volume_sum['ventilation_negociant'],
    marker=dict(size=6)  
))

fig_volumes.update_layout(
    title=f"Ventilation des volumes totaux par négociants pour {vin_selected_display} ({millesime_selected}, {format_selected})",
    xaxis_title="Date",
    yaxis_title="Quantité totale",
    hovermode="x",
    yaxis=dict(range=[0, df_volume_sum['quantite_totale'].max() + 10]),
    xaxis=dict(tickmode='auto', nticks=10))  

st.plotly_chart(fig_volumes)