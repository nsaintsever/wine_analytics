import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import matplotlib.colors as mcolors
from functools import lru_cache
import requests
from bs4 import BeautifulSoup
import yfinance as yf  # Import yfinance for exchange rates

st.set_page_config(page_title="Wine Analytics", page_icon="🍇", layout="wide")

st.markdown(
    """
    <style>
    /* Scale down the content to 70% */
    .main-content {
        transform: scale(0.8); 
        transform-origin: top left; 
    }
    /* Optional: Adjust width for centered alignment */
    section.main > div {
        max-width: 80%; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to get live GBP to EUR conversion rate from Boursorama
def get_conversion_rate_boursorama():
    url = "https://www.boursorama.com/bourse/devises/taux-de-change-livresterling-euro-GBP-EUR/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find and retrieve the last element with the specified class
    elements = soup.find_all("span", class_="c-instrument c-instrument--last")
    if elements:
        rate_text = elements[-1].text.replace(',', '.').strip()
        try:
            return float(rate_text)
        except ValueError:
            st.error("Erreur lors de la conversion du taux de change.")
    return None

# Get the live exchange rate
live_conversion_rate = get_conversion_rate_boursorama()
if live_conversion_rate is None:
    live_conversion_rate = 1.0  # Fallback to 1.0 if fetching fails
    st.warning("Taux de conversion GBP -> EUR par défaut utilisé (1.0).")

# Download historical exchange rates via yfinance
ticker = 'GBPEUR=X'
fx_data = yf.download(ticker, start='2024-01-01')
fx_data.reset_index(inplace=True)

# Create exchange rate series
fx_series = fx_data.set_index('Date')['Close']

# For the first page, use the latest exchange rate from yfinance
if fx_series.empty:
    conversion_rate = 1.1701
else:
    conversion_rate = fx_series.iloc[-1]


# Load data
df = pd.read_csv("lacoste_sub_df.csv", sep=',', quotechar='"')
df['date'] = pd.to_datetime(df['date'], format='ISO8601')
df['millesime'] = pd.to_numeric(df['millesime']).astype(int)
df['quantite_totale'] = df['per'] * df['quantite']

# Sidebar navigation and settings
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sélection page : ", ("Analyse Château - Prix & Volume", "Analyse Vin - Prix & Volume"))
st.sidebar.text("")
st.sidebar.title("Paramètres")

vin_display_names = {}
for vin_clean in df['vin_clean'].unique():
    associated_vins = df[df['vin_clean'] == vin_clean]['vin']
    nicest_name = max(associated_vins, key=lambda x: (x.count(','), len(x)))
    vin_display_names[vin_clean] = nicest_name

display_wine_options = list(vin_display_names.values())

# Define color maps globally
yellow_cmap = mcolors.LinearSegmentedColormap.from_list("yellow_gradient", ["#FFF7AE", "#FFC300", "#D4A017"])
red_cmap = mcolors.LinearSegmentedColormap.from_list("red_gradient", ["#FFE5E5", "#FF4D4D", "#800000"])
blue_cmap = mcolors.LinearSegmentedColormap.from_list("blue_gradient", ["#E0F7FA", "#0288D1", "#01579B"])  # For future use

# Define the add_bars_to_fig function globally
def add_bars_to_fig_with_vintage_coloring(fig, data, y_col, hover_template, title, yaxis_title, yellow_cmap, blue_cmap, timeframe_freq):
    # Define the vintage color mapping
    vintage_colors = {
        1980: 'blue', 1981: 'blue', 1983: 'blue', 1984: 'blue', 1985: 'blue', 1986: 'blue', 1987: 'blue', 
        1988: 'blue', 1991: 'blue', 1992: 'blue', 1993: 'blue', 1994: 'blue', 1995: 'blue', 1997: 'blue', 
        1998: 'blue', 1999: 'blue', 2001: 'blue', 2002: 'blue', 2004: 'blue', 2006: 'blue', 2007: 'blue', 
        2008: 'blue', 2011: 'blue', 2012: 'blue', 2013: 'blue', 2014: 'blue', 2017: 'blue', 2021: 'blue', 
        2023: 'blue'
    }

    unique_timeframes = {}
    x_position = 0
    centered_x_vals = []
    vintage_labels = []

    for vintage, group in data.groupby('millesime'):
        num_timeframes = len(group)
        x_values = [x_position + i for i in range(num_timeframes)]
        y_values = group[y_col].tolist()
        
        # Use `blue_cmap` if vintage is specified as blue, else `yellow_cmap`
        cmap = blue_cmap if vintage in vintage_colors else yellow_cmap
        colors = [mcolors.to_hex(cmap(i / (num_timeframes - 1))) for i in range(num_timeframes)] if num_timeframes > 1 else [mcolors.to_hex(cmap(0.5))]

        fig.add_trace(go.Bar(
            x=x_values,
            y=y_values,
            marker=dict(color=colors),
            showlegend=False,
            hovertemplate=hover_template
        ))

        centered_x_vals.append(np.mean(x_values))
        vintage_labels.append(int(vintage))

        for i, row in enumerate(group.itertuples()):
            start_date = row.date
            if timeframe_freq == "7D":
                end_date = start_date + pd.Timedelta(days=6)
            elif timeframe_freq == "14D":
                end_date = start_date + pd.Timedelta(days=13)
            elif timeframe_freq == "30D":
                end_date = start_date + pd.Timedelta(days=29)
            timeframe_label = f"{start_date.date()} - {end_date.date()}"

            if timeframe_label not in unique_timeframes:
                unique_timeframes[timeframe_label] = colors[i]

        x_position += num_timeframes + 2

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Vintage Year",
            tickvals=centered_x_vals,
            ticktext=vintage_labels,
            tickangle=-45,
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(title=yaxis_title, automargin=True, autorange=True, fixedrange=False),
        barmode='group'
    )

    for label, color in unique_timeframes.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=color),
            legendgroup=label,
            showlegend=True,
            name=label
        ))

    return fig


def add_bars_to_fig_simple(fig, data, y_col, hover_template, title, yaxis_title, red_cmap, timeframe_freq):
    unique_timeframes = {}
    x_position = 0
    centered_x_vals = []
    vintage_labels = []

    for vintage, group in data.groupby('millesime'):
        num_timeframes = len(group)
        x_values = [x_position + i for i in range(num_timeframes)]
        y_values = group[y_col].tolist()

        # Use `red_cmap` for all bars
        colors = [mcolors.to_hex(red_cmap(i / (num_timeframes - 1))) for i in range(num_timeframes)] if num_timeframes > 1 else [mcolors.to_hex(red_cmap(0.5))]

        fig.add_trace(go.Bar(
            x=x_values,
            y=y_values,
            marker=dict(color=colors),
            showlegend=False,
            hovertemplate=hover_template
        ))

        centered_x_vals.append(np.mean(x_values))
        vintage_labels.append(int(vintage))

        # Create timeframe labels for the legend
        for i, row in enumerate(group.itertuples()):
            start_date = row.date
            if timeframe_freq == "7D":
                end_date = start_date + pd.Timedelta(days=6)
            elif timeframe_freq == "14D":
                end_date = start_date + pd.Timedelta(days=13)
            elif timeframe_freq == "30D":
                end_date = start_date + pd.Timedelta(days=29)
            timeframe_label = f"{start_date.date()} - {end_date.date()}"

            if timeframe_label not in unique_timeframes:
                unique_timeframes[timeframe_label] = colors[i]

        x_position += num_timeframes + 2

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Vintage Year",
            tickvals=centered_x_vals,
            ticktext=vintage_labels,
            tickangle=-45,
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(title=yaxis_title, automargin=True, autorange=True, fixedrange=False),
        barmode='group'
    )

    # Add legend items for each unique timeframe
    for label, color in unique_timeframes.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=color),
            legendgroup=label,
            showlegend=True,
            name=label
        ))

    return fig


if page == "Analyse Château - Prix & Volume":

    selected_display_name = st.sidebar.selectbox("Choisir un vin", display_wine_options)
    selected_wine = next(key for key, value in vin_display_names.items() if value == selected_display_name)

    format_options = {vin: df[df['vin_clean'] == vin]['format'].unique() for vin in df['vin_clean'].unique()}
    selected_format = st.sidebar.selectbox("Choisir un format", format_options[selected_wine])

    timeframe_options = {"1 Semaine": "7D", "2 Semaines": "14D", "1 Mois": "30D"}
    selected_timeframe = st.sidebar.selectbox("Choisir un *timeframe*", list(timeframe_options.keys()), index=1)
    timeframe_freq = timeframe_options[selected_timeframe]

    # Page content and processing
    st.title("Analyse Château - Prix & Volume")

    # Filter and prepare data
    filtered_df = df[(df['vin_clean'] == selected_wine) & (df['format'] == selected_format)].copy()
    filtered_df['total_quantity_available'] = filtered_df['per'] * filtered_df['quantite']
    
    # Convert GBP to EUR prices using the latest exchange rate from yfinance
    filtered_df['prix_eur_equiv'] = filtered_df['prix_unitaire'] * conversion_rate

    # Apply timeframe adjustments
    if timeframe_freq == '7D':
        filtered_df['timeframe_start_date'] = filtered_df['date'] - pd.to_timedelta(filtered_df['date'].dt.weekday, unit='d')
    elif timeframe_freq == '14D':
        first_date = filtered_df['date'].min()
        days_since_first = (filtered_df['date'] - first_date).dt.days
        period_index = days_since_first // 14
        filtered_df['timeframe_start_date'] = first_date + period_index * pd.Timedelta(days=14)
    elif timeframe_freq == '30D':
        filtered_df['timeframe_start_date'] = filtered_df['date'].dt.to_period('M').dt.start_time
    else:
        filtered_df['timeframe_start_date'] = filtered_df['date']
        
    # Calculate minimum EUR price per timeframe and vintage
    min_price_per_timeframe = (
        filtered_df.groupby(['millesime', 'timeframe_start_date'])
        .agg({'prix_eur_equiv': 'min'})
        .reset_index()
    )

    # Extract date where minimum EUR price occurs
    min_price_rows = pd.merge(
        filtered_df,
        min_price_per_timeframe,
        left_on=['millesime', 'timeframe_start_date', 'prix_eur_equiv'],
        right_on=['millesime', 'timeframe_start_date', 'prix_eur_equiv']
    )

    min_price_rows['date'] = pd.to_datetime(min_price_rows['date'])

    
    # Sum total quantity on those dates for each timeframe
    min_price_totals = (
        min_price_rows.groupby(['millesime', 'date'])
        .agg({'total_quantity_available': 'max'})
        .reset_index()
    )
    
    # Rename columns for plotting
    windowed_min_prices = min_price_per_timeframe.rename(columns={'timeframe_start_date': 'date'})
    min_price_totals = min_price_totals.rename(columns={'timeframe_start_date': 'date'})

    # Plot minimum EUR equivalent price using yellow colormap
    fig_price = go.Figure()

    fig_price = add_bars_to_fig_with_vintage_coloring(
        fig_price,
        windowed_min_prices,
        'prix_eur_equiv',
        "Prix - EUR: %{y}<extra></extra>",
        f"Meilleure offre par {selected_timeframe} pour {selected_display_name} - {selected_format}",
        "Meilleure Offre (EUR - €)",
        yellow_cmap,  # Use yellow colormap for vintages without specified colors
        blue_cmap,    # Use blue colormap for vintages specified as blue
        timeframe_freq
    )


    fig_price.update_layout(
        yaxis=dict(autorange = True, fixedrange=False)
    )
    st.plotly_chart(fig_price)


    
    
    grouped_volumes = filtered_df.groupby(["millesime", "date"]).agg({"total_quantity_available": "sum"}).sort_values("date").reset_index()

    # 2. Définir la fonction pour assigner 'timeframe_start_date'
    def assign_timeframe_start_date(dates, freq):
        if freq == '7D':
            # Assignation du début de la semaine (par exemple, lundi)
            return dates - pd.to_timedelta(dates.dt.weekday, unit='d')
        elif freq == '14D':
            # Assignation par période de 14 jours à partir de la première date
            first_date = dates.min()
            return first_date + pd.to_timedelta((dates - first_date).dt.days // 14 * 14, unit='d')
        elif freq == '30D':
            # Assignation au début du mois
            return dates.dt.to_period('M').dt.start_time
        else:
            # Si aucune fréquence spécifique, assigner la date elle-même
            return dates

    # 3. Ajouter la colonne 'timeframe_start_date'
    grouped_volumes['timeframe_start_date'] = assign_timeframe_start_date(grouped_volumes['date'], timeframe_freq)

    # 4. Grouper par 'millesime' et 'timeframe_start_date' pour obtenir les totaux par timeframe
    grouped_timeframes = grouped_volumes.groupby(['millesime', 'timeframe_start_date']).agg({'total_quantity_available': 'max'}).reset_index()

    # 5. Renommer 'timeframe_start_date' en 'date' pour compatibilité avec les fonctions de plotting
    grouped_timeframes = grouped_timeframes.rename(columns={'timeframe_start_date': 'date'})

    # 6. Utiliser 'grouped_timeframes' pour tracer fig_quantity
    fig_quantity = go.Figure()

    fig_quantity = add_bars_to_fig_simple(
        fig_quantity,
        grouped_timeframes,
        'total_quantity_available',
        "Volumes visibles : %{y}<extra></extra>",
        f"Volume visibles par {selected_timeframe} pour {selected_display_name} - {selected_format}",
        "Total Volume Visible",
        red_cmap,  # Utilisez red_cmap si vous voulez conserver votre colormap rouge
        timeframe_freq
    )

    fig_quantity.update_layout(
        yaxis=dict(autorange=True, fixedrange=False),
        xaxis_title="Début du Timeframe",
        yaxis_title="Total Quantity Available",
        title=f"Volumes visibles par {selected_timeframe} pour {selected_display_name} - {selected_format}"
    )
    st.plotly_chart(fig_quantity)
    
    
    
    
    
    
        
elif page == "Analyse Vin - Prix & Volume":
    # Parameters for Page 2
    df['millesime'] = df['millesime'].fillna(0).astype(int)
    millesimes_sorted = sorted(df['millesime'].unique(), reverse=True)

    def get_longest_vin(vin_clean_group):
        return max(vin_clean_group, key=len)

    vin_clean_options = df.groupby('vin_clean')['vin'].apply(get_longest_vin).to_dict()
    vin_options = {v: k for k, v in vin_clean_options.items()}

    last_date_in_data = df['date'].max()

    vin_selected_display = st.sidebar.selectbox("Choisissez un vin", display_wine_options)
    vin_selected = next((k for k, v in vin_display_names.items() if v == vin_selected_display), None)


    millesime_selected = st.sidebar.selectbox("Choisissez un millésime", millesimes_sorted)
    format_options = df[(df['vin_clean'] == vin_selected) & (df['millesime'] == millesime_selected)]['format'].unique()
    format_selected = st.sidebar.selectbox("Choisissez un format", format_options, index=0)

    date_selected = st.sidebar.date_input("Choisissez une date", last_date_in_data.date(), min_value=df['date'].min(), max_value=last_date_in_data)

    # Page 2 Content
    st.title(f"Analyse Vin - Prix & Volume : {vin_selected_display} - {millesime_selected}")

    # Filter data for the selected wine, vintage, and format
    df_vin = df[(df['vin_clean'] == vin_selected) &
                (df['millesime'] == millesime_selected) &
                (df['format'] == format_selected)].copy()

    df_vin['quantite_totale'] = df_vin['per'] * df_vin['quantite']

    # Ensure we have data for every date
    full_dates = pd.date_range(start=df_vin['date'].min(), end=df_vin['date'].max())

    # Aggregate data to ensure unique ('date', 'negociant') pairs
    df_vin_agg = df_vin.groupby(['date', 'negociant']).agg({
        'prix_unitaire': 'min',  # Minimum price per negociant per date
        'quantite_totale': 'sum'
    }).reset_index()

    # Get all negociants
    all_negociants = df_vin_agg['negociant'].unique()

    # Create a DataFrame with all combinations of dates and negociants
    all_dates_negociants = pd.MultiIndex.from_product(
        [full_dates, all_negociants],
        names=['date', 'negociant']
    ).to_frame(index=False)

    # Merge df_vin_agg onto all_dates_negociants
    df_vin_agg_full = all_dates_negociants.merge(df_vin_agg, on=['date', 'negociant'], how='left')

    # Convert 'date' column to datetime if necessary
    if not pd.api.types.is_datetime64_any_dtype(df_vin_agg_full['date']):
        df_vin_agg_full['date'] = pd.to_datetime(df_vin_agg_full['date'])

    # Sort by 'negociant' and 'date' to prepare for forward fill
    df_vin_agg_full = df_vin_agg_full.sort_values(['negociant', 'date'])

    # Forward-fill 'prix_unitaire' per negociant
    df_vin_agg_full['prix_unitaire'] = df_vin_agg_full.groupby('negociant')['prix_unitaire'].ffill()

    # Forward-fill 'quantite_totale' per negociant with a limit of 5 days
    df_vin_agg_full['quantite_totale'] = df_vin_agg_full.groupby('negociant')['quantite_totale'].ffill(limit=5)

    # Fill remaining NaNs in 'quantite_totale' with zero (sold out)
    df_vin_agg_full['quantite_totale'] = df_vin_agg_full['quantite_totale'].fillna(0)

    # Set 'prix_unitaire' to NaN where 'quantite_totale' is zero
    df_vin_agg_full.loc[df_vin_agg_full['quantite_totale'] == 0, 'prix_unitaire'] = np.nan

    # Map historical exchange rates to df_vin_agg_full
    # Ensure 'date' is datetime and set to midnight to match fx_series index
    df_vin_agg_full['date'] = pd.to_datetime(df_vin_agg_full['date']).dt.normalize()

    # Reindex fx_series to include all dates needed and forward-fill missing rates
    all_dates_needed = pd.date_range(start=df_vin_agg_full['date'].min(), end=df_vin_agg_full['date'].max())
    fx_series_full = fx_series.reindex(all_dates_needed, method='ffill')

    # Map exchange rates to df_vin_agg_full
    df_vin_agg_full['conversion_rate'] = df_vin_agg_full['date'].map(fx_series_full)

    # Handle any remaining NaNs in 'conversion_rate'
    df_vin_agg_full['conversion_rate'] = df_vin_agg_full['conversion_rate'].fillna(method='ffill').fillna(method='bfill')

    # Compute the minimum price per date (ignore NaNs)
    df_min_price = df_vin_agg_full.groupby('date').agg({
        'prix_unitaire': lambda x: x.min(skipna=True),
        'conversion_rate': 'first'
    }).reset_index()
    df_min_price['prix_eur_min'] = df_min_price['prix_unitaire'] * df_min_price['conversion_rate']

    # Compute total quantity per date
    df_total_quantity = df_vin_agg_full.groupby('date').agg({
        'quantite_totale': 'sum'
    }).reset_index()

    # Merge the dataframes
    df_stats = pd.merge(df_min_price, df_total_quantity, on='date')

    # Sort by date
    df_stats = df_stats.sort_values('date')

    # Prepare the hover text with details per négociant
    df_quantities = df_vin_agg_full.groupby(['date', 'negociant']).agg({
        'quantite_totale': 'sum'
    }).reset_index()

    # Function to create hover text
    def create_hover_text(group):
        date = group.name
        total_volume = group['quantite_totale'].sum()
        hover_text = f"<b>Date:</b> {date:%Y-%m-%d}<br><b>Volume Total:</b> {total_volume} bouteilles<br><br><b>Détail par négociant:</b><br>"
        for _, row in group.iterrows():
            hover_text += f"<b>{row['negociant']}:</b> {row['quantite_totale']}<br>"
        return hover_text

    # Create hover texts
    hover_texts = df_quantities.groupby('date').apply(create_hover_text)

    # Merge hover texts into df_stats
    df_stats = df_stats.merge(hover_texts.rename('hover_text'), left_on='date', right_index=True, how='left')

    # Add CSS for vertical line between columns (if needed)
    st.markdown(
        """
        <style>
        div[data-testid="column"]:nth-child(1) {
            position: relative;
            padding-right: 300px;
        }
        div[data-testid="column"]:nth-child(1)::after {
            content: '';
            position: absolute;
            top: 0;
            right: 100px; 
            width: 1px;
            height: 100%;
            background-color: white;
        }
        div[data-testid="column"]:nth-child(2) {
            padding-left: -50px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create two columns for the first two components
    col1, col2 = st.columns([2, 1])

    with col1:
        # Plot the minimum price over time (GBP and EUR)
        st.subheader(f"Évolution de la meilleure offre pour {vin_selected_display} - {millesime_selected}")

        fig_price = go.Figure()

        # Trace for GBP Price
        fig_price.add_trace(go.Scatter(
            x=df_stats['date'],
            y=df_stats['prix_unitaire'],
            mode='lines+markers',
            name="Prix Min GBP",
            line=dict(color='goldenrod'),
            hovertemplate=(
                '<b>Date:</b> %{x|%Y-%m-%d}<br>'
                '<b>Prix Min GBP:</b> £%{y:.2f}<extra></extra>'
            )
        ))

        # Trace for EUR Price
        fig_price.add_trace(go.Scatter(
            x=df_stats['date'],
            y=df_stats['prix_eur_min'],
            mode='lines+markers',
            name="Prix Min EUR",
            line=dict(color='darkred', dash='dash'),
            hovertemplate=(
                '<b>Date:</b> %{x|%Y-%m-%d}<br>'
                '<b>Prix Min EUR:</b> €%{y:.2f}<extra></extra>'
            )
        ))

        fig_price.update_layout(
            xaxis_title="Date",
            yaxis_title="Prix Min",
            hovermode="x unified",
            yaxis=dict(
                rangemode='tozero',  # Start y-axis at 0
                autorange=True,
                fixedrange=False
            )
        )

        st.plotly_chart(fig_price)

    with col2:
        # Update the quantities table
        st.subheader(f"Ventilation des stocks pour {vin_selected_display} à la date du : {date_selected}")

        date_selected_datetime = pd.to_datetime(date_selected)

        # Filter df_vin_agg_full for the selected date
        df_stock = df_vin_agg_full[df_vin_agg_full['date'] == date_selected_datetime]

        # Remove negociants with zero quantities
        df_stock = df_stock[df_stock['quantite_totale'] > 0]

        # Prepare the table
        df_stock = df_stock[['negociant', 'quantite_totale', 'prix_unitaire']]
        df_stock = df_stock.dropna(subset=['prix_unitaire'])  # Remove entries where price is NaN
        df_stock = df_stock.sort_values(by='prix_unitaire', ascending=True)
        df_stock['quantite_totale'] = df_stock['quantite_totale'].astype(int)
        df_stock['prix_unitaire'] = df_stock['prix_unitaire'].round(2)
        
        # Use live exchange rate from Boursorama for the EUR equivalent
        df_stock['prix_eur_equiv'] = (df_stock['prix_unitaire'] * live_conversion_rate).round(2)
        df_stock.columns = ['Négociants', 'Stocks', 'Prix - GBP', 'Prix - EUR équiv.']

        total_volume = df_stock['Stocks'].sum()
        df_stock.loc[len(df_stock.index)] = ['Total', total_volume, '', '']

        # Style the DataFrame for display
        styled_df = df_stock.style.hide(axis='index').format(
            {'Prix - GBP': lambda x: f"£{x:.2f}" if isinstance(x, (int, float)) else x,
             'Prix - EUR équiv.': lambda x: f"€{x:.2f}" if isinstance(x, (int, float)) else x}
        ).set_table_styles([
            {'selector': 'th.col_heading.level0', 'props': [('text-align', 'center'), ('font-weight', 'bold')]},
            {'selector': 'td', 'props': [('text-align', 'center')]},
            {'selector': 'th.col_heading.level0:nth-child(1)', 'props': [('text-align', 'left')]},
            {'selector': 'td.col0', 'props': [('text-align', 'left')]}
        ])

        # CSS to center the table
        st.markdown("""
            <style>
            .center-table {
                display: flex;
                justify-content: center;
                align-items: center;
                margin: auto;
                padding-top: 50px; 
            }
            .center-table table {
                border-collapse: collapse;
                margin: auto;
            }
            .center-table th {
                background-color: #59040c !important; 
                color: white !important; 
                font-weight: bold;
                padding: 10px;
            }
            .center-table td {
                padding: 10px;
                text-align: left;
            }
            </style>
            """, unsafe_allow_html=True)

        # Display the table within the centered div
        st.markdown('<div class="center-table">' + styled_df.to_html() + '</div>', unsafe_allow_html=True)

    # Below the two columns, plot the evolution of total volumes
    st.subheader(f"Évolution des volumes totaux dans le temps pour {vin_selected_display} - {millesime_selected}")

    # Plot the volumes with detailed hover
    fig_volumes = go.Figure()

    fig_volumes.add_trace(go.Scatter(
        x=df_stats['date'],
        y=df_stats['quantite_totale'],
        mode='lines+markers',
        name="Volume total",
        line=dict(color='darkgoldenrod'),
        hovertext=df_stats['hover_text'],  # Use the custom hover text
        hoverinfo='text'  # Specify that only the text is shown
    ))

    fig_volumes.update_layout(
        xaxis_title="Date",
        yaxis_title="Quantité totale",
        hovermode="x unified",
        yaxis=dict(
            rangemode='tozero',  # Start y-axis at 0
            autorange=True,
            fixedrange=False
        ),
        xaxis=dict(tickmode='auto', nticks=10)
    )

    st.plotly_chart(fig_volumes)
