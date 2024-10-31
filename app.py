import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import matplotlib.colors as mcolors
from currency_converter import CurrencyConverter
from functools import lru_cache

st.set_page_config(page_title="🍇 Wine Analytics",layout="wide")

st.markdown(
    """
    <style>
    /* Scale down the content to 80% */
    .main-content {
        transform: scale(0.8); /* Adjust this to your preference */
        transform-origin: top left; /* Keeps content aligned with the top-left corner */
    }
    /* Optional: Adjust width for centered alignment */
    section.main > div {
        max-width: 85%; /* Adjust to control overall app width */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load data
df = pd.read_csv("lacoste_sub_df.csv", sep=',', quotechar='"')
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df['quantite_totale'] = df['per'] * df['quantite']

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sélection page : ", ("Analyse Château - Prix & Volume", "Analyse Vin - Prix & Volume"))
st.sidebar.text("")
# Sidebar parameters based on the selected page
st.sidebar.title("Paramètres")

vin_display_names = {}
for vin_clean in df['vin_clean'].unique():
    associated_vins = df[df['vin_clean'] == vin_clean]['vin']
    nicest_name = max(associated_vins, key=lambda x: (x.count(','), len(x)))
    vin_display_names[vin_clean] = nicest_name

display_wine_options = list(vin_display_names.values())

if page == "Analyse Château - Prix & Volume":

    selected_display_name = st.sidebar.selectbox("Choisir un vin", display_wine_options)
    selected_wine = next(key for key, value in vin_display_names.items() if value == selected_display_name)

    format_options = {vin: df[df['vin_clean'] == vin]['format'].unique() for vin in df['vin_clean'].unique()}
    selected_format = st.sidebar.selectbox("Choisir un format", format_options[selected_wine])

    timeframe_options = {"1 Semaine": "7D", "2 Semaines": "14D", "1 Mois": "30D"}
    selected_timeframe = st.sidebar.selectbox("Choisir un *timeframe*", list(timeframe_options.keys()), index=1)
    timeframe_freq = timeframe_options[selected_timeframe]

    # Page 1 Content
    st.title("Analyse Château - Prix & Volume")

    # Filter DataFrame
    filtered_df = df[(df['vin_clean'] == selected_wine) & (df['format'] == selected_format)].copy()
    filtered_df['total_quantity_available'] = filtered_df['per'] * filtered_df['quantite']

    if timeframe_freq == '7D':
        filtered_df['timeframe_start_date'] = filtered_df['date'] - pd.to_timedelta(filtered_df['date'].dt.weekday, unit='d')
    elif timeframe_freq == '14D':
        first_date = filtered_df['date'].min()
        days_since_first = (filtered_df['date'] - first_date).dt.days
        period_index = days_since_first // 14
        filtered_df['timeframe_start_date'] = first_date + period_index * pd.Timedelta(days=14)
    elif timeframe_freq == '30D':
        # Corrected line using .to_timestamp()
        filtered_df['timeframe_start_date'] = filtered_df['date'].dt.to_period('M').dt.start_time
    else:
        filtered_df['timeframe_start_date'] = filtered_df['date']
        
    # Calculate minimum price per timeframe and vintage
    min_price_per_timeframe = (
        filtered_df.groupby(['millesime', 'timeframe_start_date'])
        .agg({'prix_unitaire': 'min'})
        .reset_index()
    )

    # Merge to find all rows where the minimum price occurred
    min_price_rows = pd.merge(
        filtered_df,
        min_price_per_timeframe,
        on=['millesime', 'timeframe_start_date', 'prix_unitaire']
    )

    # Ensure 'date' is in datetime format
    min_price_rows['date'] = pd.to_datetime(min_price_rows['date'])

    # For each group, find the earliest date when the min price occurred
    min_price_first_dates = (
        min_price_rows.sort_values('date')
        .groupby(['millesime', 'timeframe_start_date'])
        .first()
        .reset_index()
    )

    # Extract the earliest date per group
    min_price_dates = min_price_first_dates[['millesime', 'timeframe_start_date', 'date']]

    # Now, for each vintage and timeframe, get all data for that earliest date
    min_price_day_data = pd.merge(
        filtered_df,
        min_price_dates,
        on=['millesime', 'timeframe_start_date', 'date']
    )

    # Sum total quantity available on those dates
    min_price_totals = (
        min_price_day_data.groupby(['millesime', 'timeframe_start_date'])
        .agg({'total_quantity_available': 'sum'})
        .reset_index()
    )

    # For plotting, rename 'timeframe_start_date' to 'date'
    windowed_min_prices = min_price_per_timeframe.rename(columns={'timeframe_start_date': 'date'})
    min_price_totals = min_price_totals.rename(columns={'timeframe_start_date': 'date'})

    # Visualization setup
    cmap = mcolors.LinearSegmentedColormap.from_list("yellow_gradient", ["#FFF7AE", "#FFC300", "#D4A017"])

    # Function to add bar traces for either chart
    def add_bars_to_fig(fig, data, y_col, hover_template, title, yaxis_title):
        unique_timeframes = {}
        x_position = 0
        centered_x_vals = []
        vintage_labels = []

        for vintage, group in data.groupby('millesime'):
            num_timeframes = len(group)
            x_values = [x_position + i for i in range(num_timeframes)]
            y_values = group[y_col].tolist()
            colors = [mcolors.to_hex(cmap(i / (num_timeframes - 1))) if num_timeframes > 1 else mcolors.to_hex(cmap(0.5)) for i in range(num_timeframes)]

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
            yaxis=dict(title=yaxis_title, automargin=True, autorange=True),
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

    # Plot minimum price chart
    fig_price = go.Figure()
    fig_price = add_bars_to_fig(
        fig_price,
        windowed_min_prices,
        'prix_unitaire',
        "Prix - GBP: %{y}<extra></extra>",
        f"Meilleure offre par {selected_timeframe} pour {selected_display_name} - {selected_format}",
        "Meilleure Offre (GBP - £)"
    )
    st.plotly_chart(fig_price)

    # Plot total volume chart
    fig_quantity = go.Figure()
    fig_quantity = add_bars_to_fig(
        fig_quantity,
        min_price_totals,
        'total_quantity_available',
        "Volumes visibles : %{y}<extra></extra>",
        f"Volume visibles par {selected_timeframe} pour {selected_display_name} - {selected_format}",
        "Total Volume Visible"
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
    vin_selected = vin_options[vin_selected_display]

    millesime_selected = st.sidebar.selectbox("Choisissez un millésime", millesimes_sorted)
    format_options = df[(df['vin_clean'] == vin_selected) & (df['millesime'] == millesime_selected)]['format'].unique()
    format_selected = st.sidebar.selectbox("Choisissez un format", format_options, index=0)

    date_selected = st.sidebar.date_input("Choisissez une date", last_date_in_data.date(), min_value=df['date'].min(), max_value=last_date_in_data)

    # Page 2 Content
    st.title(f"Analyse Vin - Prix & Volume : {vin_selected_display} - {millesime_selected}")

    # Currency Converter with caching
    currency_converter = CurrencyConverter()

    @lru_cache(maxsize=None)
    def get_conversion_rate(date, from_currency="GBP", to_currency="EUR"):
        try:
            rate = currency_converter.convert(1, from_currency, to_currency, date=date)
        except:
            rate = None
        return rate

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

    # Map conversion rates to df_vin_agg_full
    unique_dates = df_vin_agg_full['date'].dt.date.unique()
    conversion_rates = {date: get_conversion_rate(date) for date in unique_dates}

    df_vin_agg_full['conversion_rate'] = df_vin_agg_full['date'].dt.date.map(conversion_rates)
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

    # Add CSS for vertical line between columns
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

    # Create two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # Plot the minimum price over time
        st.subheader(f"Évolution de la meilleure offre pour {vin_selected_display} - {millesime_selected}")

        fig_price = go.Figure()

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
            yaxis=dict(range=[0, df_stats[['prix_unitaire', 'prix_eur_min']].max().max() + 10])
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
        df_stock.columns = ['Négociants', 'Stocks', 'Prix']

        total_volume = df_stock['Stocks'].sum()
        df_stock.loc[len(df_stock.index)] = ['Total', total_volume, '']

        # Style the DataFrame
        styled_df = df_stock.style.hide(axis='index').format(
            {'Prix': lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x}
        )

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

        # Function to generate hover text as plain text
    def generate_hover_text(date):
        df_date = df_vin_agg_full[df_vin_agg_full['date'] == date]
        df_date = df_date[df_date['quantite_totale'] > 0]
        df_date = df_date.dropna(subset=['prix_unitaire'])
        df_date = df_date.sort_values(by='quantite_totale', ascending=False)

        # Initialize the hover text with the total volume in bold
        total_volume = int(df_date['quantite_totale'].sum())
        hover_text = f"<b>Total Volume:</b> {total_volume} bouteilles<br><br><b>Détails:</b><br><br>"

        # Limit to top 10 négociants
        df_top = df_date.head(10)

        for idx, row in df_top.iterrows():
            negociant = row['negociant']
            quantity = int(row['quantite_totale'])
            price = row['prix_unitaire']
            # Wrap the negociant name in <b> for bold and use <br> for line break
            line = f"<b>{negociant}</b>: {quantity} bouteilles à £{price:.2f}"
            hover_text += line + "<br>"

        if len(df_date) > 10:
            hover_text += "...and others"

        return hover_text.strip()


    # Apply the function to create the 'hover_text' column
    df_stats['hover_text'] = df_stats['date'].apply(generate_hover_text)

    # Plot the total quantity over time with custom hover text
    st.subheader(f"Évolution des volumes totaux dans le temps pour {vin_selected_display} - {millesime_selected}")

    fig_volumes = go.Figure()

    fig_volumes.add_trace(go.Scatter(
        x=df_stats['date'],
        y=df_stats['quantite_totale'],
        mode='lines+markers',
        name="Volume total",
        line=dict(color='darkgoldenrod'),
        hovertext=df_stats['hover_text'],
        hovertemplate='<span style="white-space: pre-wrap;">%{hovertext}</span><extra></extra>',
        hoverlabel=dict(align="left"),
        marker=dict(size=6)
    ))

    fig_volumes.update_layout(
        xaxis_title="Date",
        yaxis_title="Quantité totale",
        hovermode="closest",
        yaxis=dict(range=[0, df_stats['quantite_totale'].max() + 10]),
        xaxis=dict(tickmode='auto', nticks=10)
    )

    st.plotly_chart(fig_volumes)
