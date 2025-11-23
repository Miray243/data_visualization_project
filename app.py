import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Spotify Advanced Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown("""
    <style>
    /* Main background - light blue */
    .main {
        background-color: #81B4D9;
    }

    /* Sidebar - dark navy */
    [data-testid="stSidebar"] {
        background-color: #1a237e;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }

    /* Header */
    header[data-testid="stHeader"] {
        background-color: #81B4D9;
    }

    /* Whole app background */
    .stApp {
        background-color: #81B4D9;
    }

    /* Text colors */
    .main h1, .main h2, .main h3, .main p, .main label {
        color: #0D0D0D !important;
    }

    /* Generic markdown/text */
    .stMarkdown, .stText {
        color: #0D0D0D !important;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #1DB954 !important;
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }

    /* Info boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #191414 !important;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    """Load and cache the Spotify dataset"""
    df = pd.read_csv("clean_dataset.csv")
    return df


# Load data
df = load_data()

# ============================================================
# SIDEBAR - GLOBAL FILTERS
# ============================================================
st.sidebar.title("🎵 Dashboard Controls")
st.sidebar.markdown("---")

st.sidebar.title("🎛 Global Filters")

# FILTER 1: Genre Selection (Multi-select)
selected_genres = st.sidebar.multiselect(
    "Filter by Genre:",
    options=sorted(df['track_genre'].unique()),
    default=None,
    help="Leave empty to show all genres"
)

# FILTER 2: Popularity Range (Slider)
popularity_range = st.sidebar.slider(
    "Popularity Range:",
    min_value=int(df['popularity'].min()),
    max_value=int(df['popularity'].max()),
    value=(int(df['popularity'].min()), int(df['popularity'].max())),
    help="Filter tracks by popularity score"
)

# FILTER 3: Tempo Range (BPM)
tempo_range = st.sidebar.slider(
    "Tempo Range (BPM):",
    min_value=int(df['tempo'].min()),
    max_value=int(df['tempo'].max()),
    value=(int(df['tempo'].min()), int(df['tempo'].max())),
    help="Filter tracks by tempo (beats per minute)"
)

st.sidebar.markdown("---")

# Apply Global Filters
df_filtered = df.copy()

if selected_genres:
    df_filtered = df_filtered[df_filtered['track_genre'].isin(selected_genres)]

df_filtered = df_filtered[
    (df_filtered['popularity'] >= popularity_range[0]) &
    (df_filtered['popularity'] <= popularity_range[1])
]

df_filtered = df_filtered[
    (df_filtered['tempo'] >= tempo_range[0]) &
    (df_filtered['tempo'] <= tempo_range[1])
]

# ---------- NAVY BOX: FILTERED DATA ----------
st.sidebar.markdown(
    f"""
    <div style="
        background-color:#1a237e;
        padding:10px 12px;
        border-radius:6px;
        margin-top:10px;
        color:#ffffff;
        font-size:13px;">
        📊 <span style="font-weight:600;">Filtered Data:</span>
        {len(df_filtered):,} / {len(df):,} tracks
    </div>
    """,
    unsafe_allow_html=True
)

# Check if filtered data is empty
if df_filtered.empty:
    st.warning("⚠ No data matches the selected filters. Please adjust the filter criteria.")
    st.stop()

# Genre column for other visualizations
genre_col = 'track_genre' if 'track_genre' in df.columns else None


# ============================================================
# MAIN DASHBOARD
# ============================================================

st.title("🎵 Spotify Advanced Analytics Dashboard")
st.markdown("### Comprehensive Music Data Analysis & Visualization")
st.markdown("---")

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📀 Total Tracks",
        value=f"{len(df_filtered):,}",
        delta=f"{len(df_filtered) - len(df):,}" if selected_genres or popularity_range != (
            df['popularity'].min(), df['popularity'].max()) else None
    )

with col2:
    st.metric(
        label="🎸 Genres",
        value=f"{df_filtered['track_genre'].nunique():,}"
    )

with col3:
    st.metric(
        label="🎤 Artists",
        value=f"{df_filtered['artists'].nunique():,}"
    )

with col4:
    avg_popularity = df_filtered['popularity'].mean()
    st.metric(
        label="⭐ Avg Popularity",
        value=f"{avg_popularity:.1f}"
    )

st.markdown("""
    <style>
    /* Main background - light blue */
    .main {
        background-color: #81B4D9;
    }

    /* Sidebar - dark navy */
    [data-testid="stSidebar"] {
        background-color: #1a237e;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }

    /* Header */
    header[data-testid="stHeader"] {
        background-color: #81B4D9;
    }

    /* Whole app background */
    .stApp {
        background-color: #81B4D9;
    }

    /* Text colors */
    .main h1, .main h2, .main h3, .main p, .main label {
        color: #0D0D0D !important;
    }

    /* Generic markdown/text */
    .stMarkdown, .stText {
        color: #0D0D0D !important;
    }

    /* ====== METRIC TITLE ====== */
    [data-testid="stMetricLabel"] {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
    }

    /* ====== METRIC NUMBERS ====== */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 1.8rem !important;
        font-weight: bold !important;
    }

    /* Info boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #191414 !important;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# ROW 1: FIRST 3 VISUALIZATIONS
# ============================================================

col1, col2, col3 = st.columns(3)

# ============================================================
# VISUALIZATION 1: CORRELATION HEATMAP
# ============================================================
with col1:
    st.subheader("⿡ Correlation Heatmap")

    audio_features = ['danceability', 'energy', 'loudness', 'speechiness',
                      'acousticness', 'instrumentalness', 'liveness', 'valence',
                      'tempo', 'popularity']

    corr_matrix = df_filtered[audio_features].corr()

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10, "color": "#000000"},
        colorbar=dict(
            title=dict(text="Correlation", font=dict(size=12, color="#000000")),
            tickmode="linear",
            tick0=-1,
            dtick=0.5,
            len=0.5,
            tickfont=dict(size=10, color="#000000")
        ),
        hovertemplate='<b>%{x}</b> ↔ <b>%{y}</b><br>Correlation: <b>%{z:.3f}</b><extra></extra>',
        zmid=0,
        zmin=-1,
        zmax=1
    ))

    fig_heatmap.update_layout(
        height=500,
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=11, color='#000000'),
            side='bottom'
        ),
        yaxis=dict(
            tickfont=dict(size=11, color='#000000'),
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='rgba(255, 255, 255, 0.95)',
        paper_bgcolor='rgba(255, 255, 255, 0.95)',
        margin=dict(l=10, r=10, t=10, b=10),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            font_color="#333333",
            bordercolor="#1DB954"
        ),
        dragmode='zoom'
    )

    st.plotly_chart(fig_heatmap, use_container_width=True, config={
        'scrollZoom': True,
        'displayModeBar': True
    })

    bigger_heatmap = st.checkbox("Bigger screen", value=False, key="heatmap_bigger")

# ============================================================
# VISUALIZATION 2: STAR GLYPH PLOT
# ============================================================
with col2:
    st.subheader("⿢ Star Glyph Plot")

    n_genres = 12

    top_genres = df_filtered['track_genre'].value_counts().head(n_genres).index.tolist()
    df_genres = df_filtered[df_filtered['track_genre'].isin(top_genres)]

    audio_features_glyph = ['danceability', 'energy', 'speechiness', 'acousticness',
                            'instrumentalness', 'liveness', 'valence']

    genre_features = df_genres.groupby('track_genre')[audio_features_glyph].mean().reset_index()

    scaler = MinMaxScaler()
    genre_features[audio_features_glyph] = scaler.fit_transform(genre_features[audio_features_glyph])

    fig_glyph = go.Figure()

    n_cols = 3
    n_rows = int(np.ceil(len(genre_features) / n_cols))

    colors = ['#1DB954', '#1ED760', '#535353', '#FF6B6B', '#4ECDC4',
              '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE',
              '#85C1E2', '#F8B739']

    for idx, row in genre_features.iterrows():
        genre = row['track_genre']
        values = row[audio_features_glyph].values

        row_pos = idx // n_cols
        col_pos = idx % n_cols

        center_x = col_pos * 2.5
        center_y = (n_rows - row_pos - 1) * 2.5

        n_points = len(audio_features_glyph)
        angles = [i * 2 * np.pi / n_points - np.pi / 2 for i in range(n_points)]

        circle_angles = np.linspace(0, 2 * np.pi, 100)
        circle_x = [center_x + 1.0 * np.cos(angle) for angle in circle_angles]
        circle_y = [center_y + 1.0 * np.sin(angle) for angle in circle_angles]

        fig_glyph.add_trace(go.Scatter(
            x=circle_x,
            y=circle_y,
            mode='lines',
            line=dict(color='lightgray', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))

        x_points = [center_x + values[i] * 1.0 * np.cos(angles[i]) for i in range(n_points)]
        y_points = [center_y + values[i] * 1.0 * np.sin(angles[i]) for i in range(n_points)]
        x_points.append(x_points[0])
        y_points.append(y_points[0])

        fig_glyph.add_trace(go.Scatter(
            x=x_points,
            y=y_points,
            fill='toself',
            fillcolor=colors[idx % len(colors)],
            opacity=0.6,
            line=dict(color=colors[idx % len(colors)], width=2),
            name=genre,
            mode='lines',
            hovertemplate='<b>' + genre.upper() + '</b><br><br>' +
                          '<br>'.join([f'{audio_features_glyph[i]}: {values[i]:.2f}'
                                       for i in range(len(audio_features_glyph))]) +
                          '<extra></extra>'
        ))

        fig_glyph.add_annotation(
            x=center_x,
            y=center_y - 1.3,
            text=f'<b>{genre.upper()}</b>',
            showarrow=False,
            font=dict(size=9, color='#000000', family='Arial Black'),
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#333333',
            borderwidth=1,
            borderpad=3
        )

    # Visual mini-legend for feature directions
    legend_center_x = n_cols * 2.5 - 1.8
    legend_center_y = n_rows * 2.5 - 0.5
    legend_radius = 0.6

    n_points = len(audio_features_glyph)
    angles = [i * 2 * np.pi / n_points - np.pi / 2 for i in range(n_points)]

    legend_values = [0.8] * n_points
    legend_x = [legend_center_x + legend_values[i] * legend_radius * np.cos(angles[i]) for i in range(n_points)]
    legend_y = [legend_center_y + legend_values[i] * legend_radius * np.sin(angles[i]) for i in range(n_points)]
    legend_x.append(legend_x[0])
    legend_y.append(legend_y[0])

    fig_glyph.add_trace(go.Scatter(
        x=legend_x,
        y=legend_y,
        fill='toself',
        fillcolor='rgba(29, 185, 84, 0.3)',
        opacity=0.8,
        line=dict(color='#1DB954', width=2),
        mode='lines',
        showlegend=False,
        hoverinfo='skip'
    ))

    legend_circle_angles = np.linspace(0, 2 * np.pi, 100)
    legend_circle_x = [legend_center_x + legend_radius * np.cos(angle) for angle in legend_circle_angles]
    legend_circle_y = [legend_center_y + legend_radius * np.sin(angle) for angle in legend_circle_angles]

    fig_glyph.add_trace(go.Scatter(
        x=legend_circle_x,
        y=legend_circle_y,
        mode='lines',
        line=dict(color='gray', width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))

    feature_labels = ['Dance', 'Energy', 'Speech', 'Acoustic', 'Instru', 'Live', 'Valence']
    label_distance = 1.0

    for i, label in enumerate(feature_labels):
        label_x = legend_center_x + label_distance * np.cos(angles[i])
        label_y = legend_center_y + label_distance * np.sin(angles[i])

        fig_glyph.add_annotation(
            x=label_x,
            y=label_y,
            text=f'<b>{label}</b>',
            showarrow=False,
            font=dict(size=8, color='#000000', family='Arial Black'),
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#1DB954',
            borderwidth=1,
            borderpad=3
        )

    fig_glyph.update_layout(
        showlegend=False,
        height=500,
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-1.5, n_cols * 2.5 + 0.5]
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-2, n_rows * 2.5 + 0.5]
        ),
        plot_bgcolor='rgba(255, 255, 255, 0.95)',
        paper_bgcolor='rgba(255, 255, 255, 0.95)',
        hovermode='closest',
        margin=dict(l=10, r=10, t=10, b=10),
        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="Arial",
            font_color="#333333",
            bordercolor="#1DB954"
        ),
        dragmode='zoom'
    )

    st.plotly_chart(fig_glyph, use_container_width=True, config={
        'scrollZoom': True,
        'displayModeBar': True
    })

    bigger_glyph = st.checkbox("Bigger screen", value=False, key="glyph_bigger")

# ============================================================
# VISUALIZATION 3: ENHANCED VIOLIN PLOT
# ============================================================
with col3:
    st.subheader("⿣ Violin Plot")

    x_axis = 'track_genre'
    y_axis = 'popularity'

    df_violin = df_filtered.copy()
    top_genres_violin = df_filtered['track_genre'].value_counts().head(8).index.tolist()
    df_violin = df_filtered[df_filtered['track_genre'].isin(top_genres_violin)]

    fig_violin = go.Figure()

    colors_violin = ['#1DB954', '#1ED760', '#FF6B6B', '#4ECDC4', '#45B7D1',
                     '#FFA07A', '#98D8C8', '#F7DC6F']

    categories = df_violin[x_axis].unique()

    show_outliers = st.session_state.get('show_outliers', True)
    show_all_points = st.session_state.get('show_all_points', False)

    for idx, category in enumerate(categories):
        category_data = df_violin[df_violin[x_axis] == category][y_axis]

        if show_all_points:
            points_mode = 'all'
            point_size = 3
        elif show_outliers:
            points_mode = 'outliers'
            point_size = 5
        else:
            points_mode = False
            point_size = 4

        fig_violin.add_trace(go.Violin(
            y=category_data,
            name=str(category),
            box_visible=True,
            meanline_visible=True,
            fillcolor=colors_violin[idx % len(colors_violin)],
            opacity=0.7,
            line=dict(color=colors_violin[idx % len(colors_violin)], width=1.5),
            points=points_mode,
            pointpos=-0.5,
            jitter=0.3,
            scalemode='width',
            width=0.5,
            marker=dict(size=point_size, opacity=0.6),
            hovertemplate='<b>' + str(category) + '</b><br>Value: <b>%{y:.2f}</b><br>' +
                          f'<i>Mean: {category_data.mean():.2f}</i><br>' +
                          f'<i>Median: {category_data.median():.2f}</i><extra></extra>'
        ))

    fig_violin.update_layout(
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(255, 255, 255, 0.95)',
        paper_bgcolor='rgba(255, 255, 255, 0.95)',
        font=dict(size=11, color='#000000'),
        hovermode='closest',
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=11, color='#000000')
        ),
        yaxis=dict(
            gridcolor='#e0e0e0',
            gridwidth=0.5,
            tickfont=dict(size=11, color='#000000')
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="Arial",
            font_color="#333333",
            bordercolor="#1DB954"
        ),
        dragmode='zoom'
    )

    st.plotly_chart(fig_violin, use_container_width=True, config={
        'scrollZoom': True,
        'displayModeBar': True
    })

    col_a, col_b, col_c = st.columns([2, 2, 2])

    with col_a:
        st.checkbox("Show Outliers", value=True, key='show_outliers')

    with col_b:
        st.checkbox("Show All Points", value=False, key='show_all_points')

    with col_c:
        bigger_violin = st.checkbox("Bigger", value=False, key="violin_bigger")

st.markdown("---")

# ============================================================
# BIGGER SCREEN VIEWS FOR ROW 1
# ============================================================

# BIGGER HEATMAP
if bigger_heatmap:
    st.markdown("### ⿡ Correlation Heatmap – Bigger Screen View")

    fig_heatmap_big = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 18, "color": "#000000"},
        colorbar=dict(
            title=dict(text="Correlation", font=dict(size=16)),
            tickmode="linear",
            tick0=-1,
            dtick=0.5,
            len=0.7,
            tickfont=dict(size=14)
        ),
        hovertemplate='<b>%{x}</b> ↔ <b>%{y}</b><br>Correlation: <b>%{z:.3f}</b><extra></extra>',
        zmid=0,
        zmin=-1,
        zmax=1
    ))

    fig_heatmap_big.update_layout(
        height=800,
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=16, color='#000000'),
            side='bottom'
        ),
        yaxis=dict(
            tickfont=dict(size=16, color='#000000'),
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='rgba(255, 255, 255, 0.95)',
        paper_bgcolor='rgba(255, 255, 255, 0.95)',
        margin=dict(l=60, r=60, t=60, b=60),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial",
            font_color="#333333",
            bordercolor="#1DB954"
        ),
        dragmode='zoom'
    )

    st.plotly_chart(fig_heatmap_big, use_container_width=True, config={
        'scrollZoom': True,
        'displayModeBar': True
    })
    st.markdown("---")

# BIGGER STAR GLYPH
if bigger_glyph:
    st.markdown("### ⿢ Star Glyph Plot – Bigger Screen View")

    fig_glyph_big = go.Figure()

    n_cols_big = 4
    n_rows_big = int(np.ceil(len(genre_features) / n_cols_big))

    for idx, row in genre_features.iterrows():
        genre = row['track_genre']
        values = row[audio_features_glyph].values

        row_pos = idx // n_cols_big
        col_pos = idx % n_cols_big

        center_x = col_pos * 3.0
        center_y = (n_rows_big - row_pos - 1) * 3.0

        n_points = len(audio_features_glyph)
        angles = [i * 2 * np.pi / n_points - np.pi / 2 for i in range(n_points)]

        circle_angles = np.linspace(0, 2 * np.pi, 100)
        circle_x = [center_x + 1.2 * np.cos(angle) for angle in circle_angles]
        circle_y = [center_y + 1.2 * np.sin(angle) for angle in circle_angles]

        fig_glyph_big.add_trace(go.Scatter(
            x=circle_x,
            y=circle_y,
            mode='lines',
            line=dict(color='lightgray', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))

        x_points = [center_x + values[i] * 1.2 * np.cos(angles[i]) for i in range(n_points)]
        y_points = [center_y + values[i] * 1.2 * np.sin(angles[i]) for i in range(n_points)]
        x_points.append(x_points[0])
        y_points.append(y_points[0])

        fig_glyph_big.add_trace(go.Scatter(
            x=x_points,
            y=y_points,
            fill='toself',
            fillcolor=colors[idx % len(colors)],
            opacity=0.6,
            line=dict(color=colors[idx % len(colors)], width=3),
            name=genre,
            mode='lines',
            hovertemplate='<b>' + genre.upper() + '</b><br><br>' +
                          '<br>'.join([f'{audio_features_glyph[i]}: {values[i]:.2f}'
                                       for i in range(len(audio_features_glyph))]) +
                          '<extra></extra>'
        ))

        fig_glyph_big.add_annotation(
            x=center_x,
            y=center_y - 1.6,
            text=f'<b>{genre.upper()}</b>',
            showarrow=False,
            font=dict(size=14, color='#000000', family='Arial Black'),
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#333333',
            borderwidth=2,
            borderpad=4
        )

    # Legend for the bigger view
    legend_center_x = n_cols_big * 3.0 - 2.2
    legend_center_y = n_rows_big * 3.0 - 0.6
    legend_radius = 0.8

    legend_values = [0.8] * n_points
    legend_x = [legend_center_x + legend_values[i] * legend_radius * np.cos(angles[i]) for i in range(n_points)]
    legend_y = [legend_center_y + legend_values[i] * legend_radius * np.sin(angles[i]) for i in range(n_points)]
    legend_x.append(legend_x[0])
    legend_y.append(legend_y[0])

    fig_glyph_big.add_trace(go.Scatter(
        x=legend_x,
        y=legend_y,
        fill='toself',
        fillcolor='rgba(29, 185, 84, 0.3)',
        opacity=0.8,
        line=dict(color='#1DB954', width=3),
        mode='lines',
        showlegend=False,
        hoverinfo='skip'
    ))

    feature_labels = ['Dance', 'Energy', 'Speech', 'Acoustic', 'Instru', 'Live', 'Valence']
    label_distance = 1.3

    for i, label in enumerate(feature_labels):
        label_x = legend_center_x + label_distance * np.cos(angles[i])
        label_y = legend_center_y + label_distance * np.sin(angles[i])

        fig_glyph_big.add_annotation(
            x=label_x,
            y=label_y,
            text=f'<b>{label}</b>',
            showarrow=False,
            font=dict(size=12, color='#000000', family='Arial Black'),
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#1DB954',
            borderwidth=2,
            borderpad=4
        )

    fig_glyph_big.update_layout(
        showlegend=False,
        height=800,
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-2, n_cols_big * 3.0 + 1]
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-2, n_rows_big * 3.0 + 1]
        ),
        plot_bgcolor='rgba(255, 255, 255, 0.95)',
        paper_bgcolor='rgba(255, 255, 255, 0.95)',
        hovermode='closest',
        margin=dict(l=40, r=40, t=40, b=40),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial",
            font_color="#333333",
            bordercolor="#1DB954"
        ),
        dragmode='zoom'
    )

    st.plotly_chart(fig_glyph_big, use_container_width=True, config={
        'scrollZoom': True,
        'displayModeBar': True
    })
    st.markdown("---")

# BIGGER VIOLIN
if bigger_violin:
    st.markdown("### ⿣ Violin Plot – Bigger Screen View")

    fig_violin_big = go.Figure()

    for idx, category in enumerate(categories):
        category_data = df_violin[df_violin[x_axis] == category][y_axis]

        if show_all_points:
            points_mode = 'all'
            point_size = 5
        elif show_outliers:
            points_mode = 'outliers'
            point_size = 7
        else:
            points_mode = False
            point_size = 6

        fig_violin_big.add_trace(go.Violin(
            y=category_data,
            name=str(category),
            box_visible=True,
            meanline_visible=True,
            fillcolor=colors_violin[idx % len(colors_violin)],
            opacity=0.7,
            line=dict(color=colors_violin[idx % len(colors_violin)], width=2.5),
            points=points_mode,
            pointpos=-0.5,
            jitter=0.3,
            scalemode='width',
            width=0.6,
            marker=dict(size=point_size, opacity=0.6),
            hovertemplate='<b>' + str(category) + '</b><br>Value: <b>%{y:.2f}</b><br>' +
                          f'<i>Mean: {category_data.mean():.2f}</i><br>' +
                          f'<i>Median: {category_data.median():.2f}</i><extra></extra>'
        ))

    fig_violin_big.update_layout(
        height=800,
        showlegend=False,
        plot_bgcolor='rgba(255, 255, 255, 0.95)',
        paper_bgcolor='rgba(255, 255, 255, 0.95)',
        font=dict(size=14, color='#000000'),
        hovermode='closest',
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=16, color='#000000'),
            title=dict(text="Genre", font=dict(size=18))
        ),
        yaxis=dict(
            gridcolor='#e0e0e0',
            gridwidth=1,
            tickfont=dict(size=16, color='#000000'),
            title=dict(text="Popularity", font=dict(size=18))
        ),
        margin=dict(l=60, r=40, t=40, b=80),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial",
            font_color="#333333",
            bordercolor="#1DB954"
        ),
        dragmode='zoom'
    )

    st.plotly_chart(fig_violin_big, use_container_width=True, config={
        'scrollZoom': True,
        'displayModeBar': True
    })
    st.markdown("---")


# ============================================================
# ROW 2: NEXT 3 VISUALIZATIONS
# ============================================================

col1, col2, col3 = st.columns(3)

# ============================================================
# VISUALIZATION 4: TREEMAP
# ============================================================
fig_treemap = None

with col1:
    st.subheader("⿤ Treemap")

    # ---------- READ CONTROL VALUES FROM SESSION ----------
    use_unfiltered = st.session_state.get("treemap_clear_filters", False)
    top_genre_n = st.session_state.get("treemap_top_genre_n", 10)
    top_artist_n = st.session_state.get("treemap_top_artist_n", 10)
    detailed = st.session_state.get("treemap_detail", False)

    # --------- COMPUTE TREEMAP DATA (SMALL VIEW) ---------
    base_df = df if use_unfiltered else df_filtered

    if "track_genre" not in base_df.columns or "popularity" not in base_df.columns:
        st.error("Required columns (track_genre, popularity) not found.")
    else:
        # 1) Average popularity per genre
        genre_pop = (
            base_df.groupby("track_genre")["popularity"]
            .mean()
            .reset_index(name="avg_popularity")
            .sort_values("avg_popularity", ascending=False)
        )

        # Top N most popular genres
        top_genres = genre_pop.head(top_genre_n)["track_genre"].tolist()
        df_top_genre = base_df[base_df["track_genre"].isin(top_genres)].copy()

        # 2) Average popularity per (genre, artist)
        ga = (
            df_top_genre
            .groupby(["track_genre", "artists"])["popularity"]
            .mean()
            .reset_index(name="avg_popularity")
        )

        ga = ga.sort_values(
            ["track_genre", "avg_popularity"],
            ascending=[True, False]
        )

        # Top M artists per genre
        ga_top = ga.groupby("track_genre").head(top_artist_n)

        # Track counts
        counts = (
            df_top_genre
            .groupby(["track_genre", "artists"])
            .size()
            .reset_index(name="track_count")
        )

        # Merge average popularity + track counts
        ga_top = ga_top.merge(counts, on=["track_genre", "artists"], how="left")

        if ga_top.empty:
            st.warning("No genres/artists match the selected filters.")
        else:
            # --------- SMALL TREEMAP (simple view) ---------
            if not detailed:
                fig_treemap = px.treemap(
                    ga_top,
                    path=["track_genre", "artists"],
                    values="track_count",
                    color="avg_popularity",
                    color_continuous_scale="Viridis",
                    hover_data={"track_count": True, "avg_popularity": ':.2f'},
                    title=""
                )

            # --------- DETAILED TREEMAP (Genre → Artist → Track) ---------
            else:
                # Rows for selected (genre, artist)
                df_treemap_base = df_top_genre.merge(
                    ga_top[["track_genre", "artists"]],
                    on=["track_genre", "artists"],
                    how="inner"
                )

                # Collapse duplicates to one row per track_name
                # (if album differs, take the album of the most popular record)
                df_treemap_base = df_treemap_base.sort_values("popularity", ascending=False)

                df_grouped = (
                    df_treemap_base
                    .groupby(
                        ["track_genre", "artists", "track_name"],
                        as_index=False
                    )
                    .agg(
                        popularity=("popularity", "mean"),
                        album_name=("album_name", "first")  # album of the most popular record
                    )
                )

                # Take the top 400 tracks by popularity (to avoid overcrowding)
                df_small = (
                    df_grouped
                    .sort_values("popularity", ascending=False)
                    .head(400)
                    .copy()
                )

                # Area = 1 per track
                df_small["track_count"] = 1

                # Detailed treemap: Genre → Artist → Track (album appears in hover)
                fig_treemap = px.treemap(
                    df_small,
                    path=["track_genre", "artists", "track_name"],
                    values="track_count",
                    color="popularity",
                    color_continuous_scale="Viridis",
                    hover_data={"album_name": True, "popularity": ':.2f'},
                    title=""
                )

            # --------- SHARED LAYOUT SETTINGS ---------
            fig_treemap.update_layout(
                width=500,
                height=500,
                margin=dict(l=0, r=0, t=0, b=0),
                coloraxis_colorbar=dict(
                    thickness=10,
                    len=0.90,
                    x=1.10,
                    xpad=0
                )
            )

            # Shift domain slightly to the left to give colorbar space
            if fig_treemap.data and hasattr(fig_treemap.data[0], "domain"):
                fig_treemap.data[0].domain = dict(
                    x=[0.0, 0.95],
                    y=[0.0, 1.0]
                )

        st.plotly_chart(
            fig_treemap,
            use_container_width=True,
            config={
                'scrollZoom': True,
                'displayModeBar': True
            }
        )

    # ---------- CONTROL BAR DIRECTLY UNDER THE CHART ----------
    col_a, col_b, col_c, col_d = st.columns([2, 2, 2, 2])

    with col_a:
        st.checkbox(
            "All Data",
            value=use_unfiltered,
            key="treemap_clear_filters"
        )

    with col_b:
        st.slider(
            "Top Genres",
            min_value=3,
            max_value=50,
            value=top_genre_n,
            step=1,
            key="treemap_top_genre_n"
        )

    with col_c:
        st.slider(
            "Top Artists",
            min_value=3,
            max_value=50,
            value=top_artist_n,
            step=1,
            key="treemap_top_artist_n"
        )

    with col_d:
        st.checkbox(
            "Detail",
            value=detailed,
            key="treemap_detail"
        )

    # Small checkbox (similar to Violin)
    bigger_treemap = st.checkbox(
        "Bigger screen",
        value=False,
        key="treemap_bigger"
    )

# ============================================================
# BIGGER TREEMAP VIEW (full-width, similar to Violin bigger)
# ============================================================
if st.session_state.get("treemap_bigger", False) and fig_treemap is not None:
    st.markdown("### ⿤ Treemap – Bigger Screen View")

    fig_treemap_big = go.Figure(fig_treemap)
    fig_treemap_big.update_layout(
        height=800,
        margin=dict(l=60, r=60, t=40, b=40)
    )

    st.plotly_chart(fig_treemap_big, use_container_width=True)
    st.markdown("---")


# ============================================================
# VISUALIZATION 5: SCATTER PLOT MATRIX
# ============================================================

fig_splom = None

with col2:
    st.subheader("⿥ Scatterplot Matrix")

    # Available numeric features
    numeric_candidates = [
        "popularity", "danceability", "energy", "valence",
        "tempo", "loudness", "acousticness",
        "instrumentalness", "speechiness", "duration_ms"
    ]

    numeric_cols = [c for c in numeric_candidates if c in df_filtered.columns]

    if "track_genre" not in df_filtered.columns or len(numeric_cols) < 2:
        st.error("Scatterplot matrix requires 'track_genre' and at least two numeric columns.")
        st.stop()

    # SESSION STATE VALUES
    default_dims = [c for c in ["energy", "tempo", "loudness"] if c in numeric_cols]
    selected_dims = st.session_state.get("splom_dimensions", default_dims)

    # enforce 2–3 dimensions
    selected_dims = [c for c in selected_dims if c in numeric_cols]
    if len(selected_dims) < 2:
        selected_dims = numeric_cols[:2]
    if len(selected_dims) > 3:
        selected_dims = selected_dims[:3]

    top_n_genres = st.session_state.get("splom_top_genres", 50)
    top_n_genres = int(max(5, min(top_n_genres, 114)))

    bigger_splom = st.session_state.get("splom_bigger", False)

    # DATA PREP
    top_genres = (
        df_filtered["track_genre"]
        .value_counts()
        .head(top_n_genres)
        .index
    )

    df_small = df_filtered[df_filtered["track_genre"].isin(top_genres)]
    df_mean = df_small.groupby("track_genre")[selected_dims].mean().reset_index()

    # MAIN SCATTERPLOT
    fig_splom = px.scatter_matrix(
        df_mean,
        dimensions=selected_dims,
        color="track_genre",
        symbol="track_genre",
        labels={c: c for c in selected_dims}
    )

    fig_splom.update_traces(
        diagonal_visible=False,
        marker=dict(size=6, opacity=0.8)
    )

    fig_splom.update_layout(
        width=500,
        height=500,
        margin=dict(l=10, r=140, t=10, b=10),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.03,
            font=dict(size=9),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        )
    )

    # PLOT
    st.plotly_chart(fig_splom, use_container_width=True, config={
        'scrollZoom': True,
        'displayModeBar': True
    })

    # ============================================================
    # CONTROL BAR – DIRECTLY UNDER THE CHART
    # ============================================================

    c1, c2, c3 = st.columns([3, 2, 1.5])

    with c1:
        st.multiselect(
            "Select up to 3 features",
            numeric_cols,
            default=selected_dims,
            key="splom_dimensions"
        )

    with c2:
        st.slider(
            "Number of genres",
            min_value=5,
            max_value=114,
            value=top_n_genres,
            step=1,
            key="splom_top_genres"
        )

    with c3:
        st.checkbox(
            "Bigger",
            value=bigger_splom,
            key="splom_bigger"
        )

# ============================================================
# BIGGER SCREEN VIEW
# ============================================================
if st.session_state.get("splom_bigger", False):
    st.markdown("### ⿥ Scatterplot Matrix – Bigger Screen View")

    fig_big = go.Figure(fig_splom)
    fig_big.update_layout(
        height=900,
        margin=dict(l=50, r=160, t=50, b=50)
    )

    st.plotly_chart(fig_big, use_container_width=True)
    st.markdown("---")


# ============================================================
# VISUALIZATION 6: HISTOGRAM
# ============================================================

fig_hist = None

with col3:
    st.subheader("⿦ Histogram")

    # Available numeric features
    histogram_features = {
        "Tempo (BPM)": "tempo",
        "Popularity": "popularity",
        "Loudness (DB)": "loudness",
        "Energy": "energy",
        "Danceability": "danceability"
    }

    available_features = {
        label: col
        for label, col in histogram_features.items()
        if col in df.columns
    }

    if not available_features:
        st.error("No numeric columns available for histogram.")
        st.stop()

    feature_labels = list(available_features.keys())
    default_feature = feature_labels[0]

    # SESSION VALUES
    selected_label = st.session_state.get("hist_feature", default_feature)
    bins = st.session_state.get("hist_bins", 40)
    bigger_hist = st.session_state.get("hist_big", False)

    # Build small histogram
    selected_col = available_features[selected_label]

    fig_hist = px.histogram(
        df_filtered,
        x=selected_col,
        nbins=bins,
        title=f"Distribution of {selected_label}"
    )

    fig_hist.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True
        }
    )

    # ==========================================================
    # CLEAN ONE-LINE CONTROL BAR
    # ==========================================================

    c1, c2, c3 = st.columns([3, 3, 2])

    with c1:
        st.selectbox(
            "Feature",
            feature_labels,
            key="hist_feature"
        )

    with c2:
        st.slider(
            "Bin count",
            min_value=10,
            max_value=80,
            value=bins,
            step=5,
            key="hist_bins"
        )

    with c3:
        st.checkbox(
            "Bigger screen",
            value=bigger_hist,
            key="hist_big"
        )

# ============================================================
# BIGGER VIEW
# ============================================================
if st.session_state.get("hist_big", False):
    st.markdown("### ⿦ Histogram – Bigger Screen View")

    fig_hist_big = go.Figure(fig_hist)
    fig_hist_big.update_layout(
        height=750,
        margin=dict(l=60, r=60, t=40, b=60)
    )

    st.plotly_chart(fig_hist_big, use_container_width=True)
    st.markdown("---")


# ============================================================
# ROW 3: LAST 3 VISUALIZATIONS
# ============================================================

col1, col2, col3 = st.columns(3)

# ============================================================
# VISUALIZATION 7: PARALLEL COORDINATES
# ============================================================

with col1:
    st.subheader("⿧ Parallel Coordinates")

    par_cols = [
        "popularity",
        "danceability",
        "energy",
        "valence",
        "tempo",
        "loudness",
        "duration_ms",
    ]
    par_cols = [c for c in par_cols if c in df_filtered.columns]

    bigger = False  # default

    if len(par_cols) < 2:
        st.warning("At least 2 numeric columns are required.")
    else:
        cols_for_sample = par_cols.copy()
        if genre_col:
            cols_for_sample.append(genre_col)

        sample = df_filtered[cols_for_sample].copy()
        if len(sample) > 1200:
            sample = sample.sample(1200, random_state=42)

        if genre_col and genre_col in sample.columns:
            sample["genre_code"] = sample[genre_col].astype("category").cat.codes
            color_arg = "genre_code"
        else:
            color_arg = "popularity"

        # ---------- SMALL VIEW (inside card) ----------
        fig_small = px.parallel_coordinates(
            sample,
            dimensions=par_cols,
            color=color_arg,
            color_continuous_scale=px.colors.sequential.Magma,
            labels={c: c.capitalize() for c in par_cols},
        )

        fig_small.update_traces(
            labelfont=dict(size=11, color="black"),
            tickfont=dict(size=9, color="black"),
        )

        fig_small.update_layout(
            coloraxis_colorbar=dict(
                title_font=dict(color="black", size=11),
                tickfont=dict(color="black", size=9),
            ),
            font=dict(size=10, color="black"),
            height=500,                                  # same height as Sunburst
            margin=dict(l=40, r=20, t=50, b=30),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        st.plotly_chart(fig_small, use_container_width=True)

        # Checkbox directly under the small chart
        bigger = st.checkbox("Bigger screen", value=False)

# ------------------------------------------------------------
#  FULL-WIDTH LARGE VIEW
# ------------------------------------------------------------
if bigger:
    st.markdown("### ⿧ Parallel Coordinates – Bigger Screen View")

    fig_big = px.parallel_coordinates(
        sample,
        dimensions=par_cols,
        color=color_arg,
        color_continuous_scale=px.colors.sequential.Magma,
        labels={c: c.capitalize() for c in par_cols},
    )

    fig_big.update_traces(
        labelfont=dict(size=20, color="black"),
        tickfont=dict(size=16, color="black"),
    )

    fig_big.update_layout(
        coloraxis_colorbar=dict(
            title_font=dict(color="black", size=18),
            tickfont=dict(color="black", size=14),
        ),
        font=dict(size=18, color="black"),
        height=800,
        margin=dict(l=60, r=40, t=80, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    st.plotly_chart(fig_big, use_container_width=True)

# ============================================================
# VISUALIZATION 8: SUNBURST CHART
# ============================================================
with col2:
    st.subheader("⿨ Sunburst Chart")

    # Default for small/normal view
    bigger_sun = False

    if genre_col is None:
        st.warning("Sunburst requires a genre column.")
    else:
        # Sample top genres and artists
        top_genres_sun = df_filtered['track_genre'].value_counts().head(8).index
        sun_df = df_filtered[df_filtered['track_genre'].isin(top_genres_sun)][
            [genre_col, "artists", "duration_ms", "popularity"]
        ].copy()

        # Aggregate to avoid too many categories
        sun_agg = sun_df.groupby([genre_col, 'artists']).agg({
            'duration_ms': 'sum',
            'popularity': 'mean'
        }).reset_index()

        # Keep top 5 artists per genre
        sun_agg = sun_agg.sort_values(['track_genre', 'popularity'],
                                      ascending=[True, False])
        sun_agg = sun_agg.groupby('track_genre').head(5)

        # ---------- SMALL (card) SUNBURST ----------
        fig_sun_small = px.sunburst(
            sun_agg,
            path=[genre_col, "artists"],
            values="duration_ms",
            color="popularity",
            color_continuous_scale=px.colors.sequential.Viridis
        )

        # Small view hover/label settings
        fig_sun_small.update_traces(
            textfont=dict(size=12),
            hoverlabel=dict(
                font=dict(
                    size=13,
                    color="black"
                )
            )
        )

        # Colorbar readability
        fig_sun_small.update_layout(
            coloraxis_colorbar=dict(
                title_font=dict(size=11, color="black"),
                tickfont=dict(size=9, color="black"),
            ),
            height=500,
            margin=dict(l=10, r=10, t=10, b=10)
        )

        st.plotly_chart(fig_sun_small, use_container_width=True)

        # Checkbox under the chart
        bigger_sun = st.checkbox("Bigger screen", value=False, key="sunburst_bigger")

# ------------------------------------------------------------
#  FULL-WIDTH BIG SUNBURST VIEW
# ------------------------------------------------------------
if genre_col is not None and bigger_sun:
    st.markdown("### ⿨ Sunburst Chart – Bigger Screen View")

    fig_sun_big = px.sunburst(
        sun_agg,
        path=[genre_col, "artists"],
        values="duration_ms",
        color="popularity",
        color_continuous_scale=px.colors.sequential.Viridis
    )

    fig_sun_big.update_traces(
        textfont=dict(size=18),
        hoverlabel=dict(
            font=dict(
                size=19,
                color="black"
            )
        )
    )

    fig_sun_big.update_layout(
        coloraxis_colorbar=dict(
            title_font=dict(size=18, color="black"),
            tickfont=dict(size=14, color="black"),
        ),
        font=dict(size=18, color="black"),
        height=800,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig_sun_big, use_container_width=True)


# ============================================================
# VISUALIZATION 9: LINE CHART
# ============================================================

if genre_col:

    # --- 1) Read current metric from session_state ---
    metric_options = ["tempo", "loudness", "duration_ms", "energy", "valence"]
    current_metric = st.session_state.get("line_metric_metric", "tempo")
    if current_metric not in metric_options:
        current_metric = "tempo"

    # --- 2) Prepare data based on selected metric ---
    line_df = df_filtered[[genre_col, current_metric, "popularity"]].copy()

    if current_metric == "tempo":
        line_df["metric_bin"] = line_df[current_metric].round()
    else:
        line_df["metric_bin"] = line_df[current_metric].round(2)

    # Top 6 genres
    top_genres_line = df_filtered["track_genre"].value_counts().head(6).index
    line_df = line_df[line_df[genre_col].isin(top_genres_line)]

    agg = (
        line_df.groupby([genre_col, "metric_bin"])["popularity"]
        .mean()
        .reset_index()
        .sort_values("metric_bin")
    )

    custom_colors = px.colors.qualitative.Bold

    # --- 3) Small chart: on the same row with others ---
    with col3:
        st.subheader("⿩ Line Chart")

        # Placeholder so the chart stays on top
        chart_placeholder = st.empty()

        # Small dashboard view
        fig_line_small = px.line(
            agg,
            x="metric_bin",
            y="popularity",
            color=genre_col,
            markers=True,
            color_discrete_sequence=custom_colors,
        )

        fig_line_small.update_traces(
            line=dict(width=2),
            marker=dict(size=4),
        )

        fig_line_small.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(size=12, color="black"),
            legend=dict(
                font=dict(size=10, color="black"),
                bgcolor="rgba(255,255,255,0.6)",
            ),
            xaxis=dict(
                title=current_metric,
                title_font=dict(size=14, color="black"),
                tickfont=dict(size=10, color="black"),
            ),
            yaxis=dict(
                title="Popularity",
                title_font=dict(size=14, color="black"),
                tickfont=dict(size=10, color="black"),
            ),
        )

        # Render the small chart
        with chart_placeholder:
            st.plotly_chart(fig_line_small, use_container_width=True)

        # --- Controls under the chart ---
        selected_metric = st.selectbox(
            "Select X-axis Metric",
            metric_options,
            index=metric_options.index(current_metric),
            key="line_metric_metric",
        )

        # Line-chart-specific bigger checkbox
        bigger_line = st.checkbox("Bigger screen", key="line_bigger")

    # --- 4) Large line chart view under the row ---
    if bigger_line:
        st.markdown("### ⿩ Line Chart – Bigger Screen View")

        fig_line_big = px.line(
            agg,
            x="metric_bin",
            y="popularity",
            color=genre_col,
            markers=True,
            color_discrete_sequence=custom_colors,
        )

        fig_line_big.update_traces(
            line=dict(width=3),
            marker=dict(size=7),
        )

        fig_line_big.update_layout(
            height=800,
            margin=dict(l=40, r=40, t=40, b=40),
            font=dict(size=15, color="black"),
            legend=dict(
                font=dict(size=14, color="black"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="black",
                borderwidth=1,
            ),
            xaxis=dict(
                title=current_metric,
                title_font=dict(size=18, color="black"),
                tickfont=dict(size=13, color="black"),
            ),
            yaxis=dict(
                title="Popularity",
                title_font=dict(size=18, color="black"),
                tickfont=dict(size=13, color="black"),
            ),
        )

        st.plotly_chart(fig_line_big, use_container_width=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #0D0D0D; padding: 20px;'>
    <p><b>🎵 Spotify Advanced Analytics Dashboard</b></p>
    <p>Built with Streamlit • Powered by Plotly • Music Data Visualization Project</p>
</div>
""", unsafe_allow_html=True)
