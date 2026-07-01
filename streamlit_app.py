import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Online Retail II - Dashboard", layout="wide")

# ===================== CACHE DATA =====================

@st.cache_data
def load_and_clean():
    df = pd.read_excel('./online_retail_II.xlsx')
    df = df.dropna(subset=['Customer ID']).copy()
    df = df.drop_duplicates()
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]
    df['Customer ID'] = df['Customer ID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['Price']
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['DayWeek'] = df['InvoiceDate'].dt.day_name()
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    return df


@st.cache_data
def build_rfm(_df):
    ref = _df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = _df.groupby('Customer ID').agg(
        Recencia=('InvoiceDate', lambda x: (ref - x.max()).days),
        Frecuencia=('Invoice', 'nunique'),
        Monetario=('TotalPrice', 'sum')
    ).reset_index()
    return rfm


@st.cache_data
def run_kmeans(_rfm, k):
    scaler = StandardScaler()
    X = _rfm[['Recencia', 'Frecuencia', 'Monetario']].values
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    _rfm = _rfm.copy()
    _rfm['Cluster_KM'] = labels
    return _rfm, scaler, km, X_scaled, labels


# ===================== LOAD =====================

df = load_and_clean()
rfm_base = build_rfm(df)

# Sidebar
st.sidebar.title("Filtros")

all_countries = ['Todos'] + sorted(df['Country'].unique().tolist())
selected_country = st.sidebar.selectbox("País", all_countries)

min_date = df['InvoiceDate'].min().date()
max_date = df['InvoiceDate'].max().date()
date_range = st.sidebar.date_input("Rango de fechas", [min_date, max_date])

k_clusters = st.sidebar.slider("Nº clusters K-Means", 2, 6, 4)

# Apply filters
df_filtered = df.copy()
if selected_country != 'Todos':
    df_filtered = df_filtered[df_filtered['Country'] == selected_country]
if len(date_range) == 2:
    df_filtered = df_filtered[
        (df_filtered['InvoiceDate'].dt.date >= date_range[0]) &
        (df_filtered['InvoiceDate'].dt.date <= date_range[1])
    ]

rfm = build_rfm(df_filtered)
rfm, scaler, km_model, X_scaled, labels = run_kmeans(rfm, k_clusters)

# Merge cluster back
df_filtered = df_filtered.merge(rfm[['Customer ID', 'Cluster_KM']], on='Customer ID', how='left')

# Train linear regression
lr = LinearRegression()
lr.fit(rfm[['Recencia', 'Frecuencia']], rfm['Monetario'])
rfm['Gasto_Predicho'] = lr.predict(rfm[['Recencia', 'Frecuencia']])
df_filtered = df_filtered.merge(rfm[['Customer ID', 'Gasto_Predicho']], on='Customer ID', how='left')

# ===================== KPIs =====================

col1, col2, col3, col4 = st.columns(4)

total_rev = df_filtered['TotalPrice'].sum()
total_cust = df_filtered['Customer ID'].nunique()
total_orders = df_filtered['Invoice'].nunique()
avg_spend = rfm['Monetario'].mean()

col1.metric("Ingresos Totales", f"£{total_rev:,.0f}")
col2.metric("Clientes", f"{total_cust:,}")
col3.metric("Pedidos", f"{total_orders:,}")
col4.metric("Gasto Promedio", f"£{avg_spend:,.0f}")

# ===================== TABS =====================

tabs = st.tabs(["EDA", "K-Means Clustering", "Predicción de Gasto"])

# ===================== TAB 1: EDA =====================

with tabs[0]:
    st.subheader("Análisis Exploratorio")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.line(
            df_filtered.groupby('YearMonth')['TotalPrice'].sum().reset_index(),
            x='YearMonth', y='TotalPrice', markers=True,
            labels={'TotalPrice': 'Ingresos (£)', 'YearMonth': 'Mes'},
            title='Ingresos Mensuales'
        )
        fig.update_traces(line_color='steelblue', line_width=2.5)
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')

    with c2:
        country_data = df_filtered.groupby('Country')['TotalPrice'].sum().reset_index()
        country_data = country_data.sort_values('TotalPrice', ascending=False).head(10)
        fig = px.bar(country_data, x='Country', y='TotalPrice', color='TotalPrice',
                     color_continuous_scale='Blues', title='Top 10 Países')
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')

    c3, c4 = st.columns(2)

    with c3:
        order_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_data = df_filtered.groupby('DayWeek')['TotalPrice'].sum().reindex(order_days).reset_index()
        fig = px.bar(day_data, x='DayWeek', y='TotalPrice', color='TotalPrice',
                     color_continuous_scale='Sunset', title='Ingresos por Día de la Semana')
        fig.update_layout(height=380)
        st.plotly_chart(fig, width='stretch')

    with c4:
        top_prods = df_filtered.groupby('Description')['TotalPrice'].sum().nlargest(10).reset_index()
        fig = px.bar(top_prods, x='TotalPrice', y='Description', orientation='h',
                     color='TotalPrice', color_continuous_scale='Greens',
                     title='Top 10 Productos')
        fig.update_layout(height=380, yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig, width='stretch')

# ===================== TAB 2: K-MEANS =====================

with tabs[1]:
    st.subheader(f"Segmentación con K-Means (k={k_clusters})")

    # Elbow + Silhouette
    K_range = range(2, 11)
    inertias = []
    from sklearn.metrics import silhouette_score
    silhouettes = []
    for k_test in K_range:
        km_test = KMeans(n_clusters=k_test, random_state=42, n_init=10)
        lb = km_test.fit_predict(X_scaled)
        inertias.append(km_test.inertia_)
        silhouettes.append(silhouette_score(X_scaled, lb))

    c5, c6 = st.columns(2)

    with c5:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode='lines+markers',
                                 marker=dict(size=8), line=dict(color='#636EFA', width=2.5),
                                 name='Inercia'))
        fig.update_layout(title='Método del Codo', height=400, xaxis_title='k',
                          yaxis_title='Inercia (WCSS)')
        st.plotly_chart(fig, width='stretch')

    with c6:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(K_range), y=silhouettes, mode='lines+markers',
                                 marker=dict(size=8, color='#EF553B'),
                                 line=dict(color='#EF553B', width=2.5),
                                 name='Silhouette'))
        fig.add_hline(y=max(silhouettes), line_dash='dash', line_color='gray')
        fig.add_vline(x=k_clusters, line_dash='dot', line_color='green')
        fig.update_layout(title='Silhouette Score', height=400, xaxis_title='k',
                          yaxis_title='Score')
        st.plotly_chart(fig, width='stretch')

    # 3D Scatter
    c7, c8 = st.columns(2)

    with c7:
        cluster_names = {0: 'Cluster A', 1: 'Cluster B', 2: 'Cluster C', 3: 'Cluster D', 4: 'Cluster E', 5: 'Cluster F'}
        palette = px.colors.qualitative.Set2[:k_clusters]

        fig = go.Figure()
        for cluster in sorted(rfm['Cluster_KM'].unique()):
            mask = rfm['Cluster_KM'] == cluster
            fig.add_trace(go.Scatter3d(
                x=rfm.loc[mask, 'Recencia'],
                y=rfm.loc[mask, 'Frecuencia'],
                z=rfm.loc[mask, 'Monetario'],
                mode='markers',
                marker=dict(size=3, opacity=0.6, color=palette[cluster % len(palette)]),
                name=cluster_names.get(cluster, f'C{cluster}')
            ))
        fig.update_layout(
            title='Clientes en espacio RFM 3D',
            height=500,
            scene=dict(
                xaxis_title='Recencia (días)',
                yaxis_title='Frecuencia (# pedidos)',
                zaxis_title='Monetario (£)'
            )
        )
        st.plotly_chart(fig, width='stretch')

    with c8:
        profile = rfm.groupby('Cluster_KM')[['Recencia', 'Frecuencia', 'Monetario']].mean().round(1)
        profile['Clientes'] = rfm.groupby('Cluster_KM').size()
        profile['%'] = (profile['Clientes'] / len(rfm) * 100).round(1)
        st.markdown("**Perfil por Cluster**")
        st.dataframe(profile)

    # Cluster distribution pie + Revenue by cluster
    c9, c10 = st.columns(2)

    with c9:
        cluster_counts = rfm['Cluster_KM'].value_counts().sort_index()
        fig = go.Figure(data=[go.Pie(
            labels=[cluster_names.get(i, f'Cluster {i}') for i in cluster_counts.index],
            values=cluster_counts.values,
            hole=0.5,
            marker_colors=palette[:len(cluster_counts)]
        )])
        fig.update_layout(title='Distribución de Clientes por Segmento', height=400)
        st.plotly_chart(fig, width='stretch')

    with c10:
        cluster_rev = rfm.groupby('Cluster_KM')['Monetario'].sum()
        fig = go.Figure(data=[go.Bar(
            x=[cluster_names.get(i, f'C{i}') for i in cluster_rev.index],
            y=cluster_rev.values,
            marker_color=palette[:len(cluster_rev)],
            text=[f'£{v:,.0f}' for v in cluster_rev.values],
            textposition='outside'
        )])
        fig.update_layout(title='Ingresos por Segmento', height=400, yaxis_title='Ingresos (£)')
        st.plotly_chart(fig, width='stretch')

# ===================== TAB 3: PREDICCIÓN =====================

with tabs[2]:
    st.subheader("Regresión Lineal: Recencia + Frecuencia → Gasto")

    coef_df = pd.DataFrame({
        'Variable': ['Intercepto', 'Recencia', 'Frecuencia'],
        'Coeficiente': [lr.intercept_, lr.coef_[0], lr.coef_[1]]
    })
    coef_df['Coeficiente'] = coef_df['Coeficiente'].round(2)
    st.markdown(f"**Ecuación:** Gasto = {lr.intercept_:.2f} + ({lr.coef_[0]:.2f} × Recencia) + ({lr.coef_[1]:.2f} × Frecuencia)")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    X_lr = rfm[['Recencia', 'Frecuencia']].values
    y_lr = rfm['Monetario'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)
    y_pred = lr.predict(X_te)

    c11, c12, c13 = st.columns(3)
    c11.metric("R² (test)", f"{r2_score(y_te, y_pred):.4f}")
    c12.metric("MAE (test)", f"£{mean_absolute_error(y_te, y_pred):.2f}")
    c13.metric("Coef. Frecuencia", f"£{lr.coef_[1]:.2f}/pedido")

    c14, c15 = st.columns(2)

    with c14:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_te, y=y_pred, mode='markers',
                                 marker=dict(opacity=0.4, color='steelblue'),
                                 name='Real vs Predicho'))
        max_v = max(y_te.max(), y_pred.max())
        min_v = min(y_te.min(), y_pred.min())
        fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v],
                                 mode='lines', line=dict(color='red', dash='dash'),
                                 name='Ideal'))
        fig.update_layout(title='Gasto Real vs Predicho (Test)', height=450,
                          xaxis_title='Real (£)', yaxis_title='Predicho (£)')
        st.plotly_chart(fig, width='stretch')

    with c15:
        size = np.abs(y_pred - y_te)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_te, y=y_pred - y_te, mode='markers',
                                 marker=dict(opacity=0.4, color='#EF553B',
                                             size=np.clip(size / size.max() * 20, 3, 20)),
                                 name='Residuos'))
        fig.add_hline(y=0, line_dash='dash', line_color='gray')
        fig.update_layout(title='Residuos (Error)', height=450,
                          xaxis_title='Real (£)', yaxis_title='Error (£)')
        st.plotly_chart(fig, width='stretch')

    # Top/Bottom predictions
    st.markdown("---")
    st.markdown("**Comparativa: Gasto Real vs Predicho**")

    rfm_display = rfm[['Customer ID', 'Recencia', 'Frecuencia', 'Monetario', 'Gasto_Predicho', 'Cluster_KM']].copy()
    rfm_display['Diferencia'] = (rfm_display['Monetario'] - rfm_display['Gasto_Predicho']).round(2)
    rfm_display.columns = ['Cliente', 'Recencia (d)', 'Frecuencia (#)', 'Gasto Real (£)',
                           'Gasto Predicho (£)', 'Cluster', 'Diferencia (£)']

    c16, c17 = st.columns(2)
    with c16:
        st.markdown("**Clientes que más gastan vs predicción**")
        st.dataframe(rfm_display.nlargest(10, 'Gasto Real (£)'))
    with c17:
        st.markdown("**Mayor subestimación (gastan más de lo predicho)**")
        st.dataframe(rfm_display.nlargest(10, 'Diferencia (£)'))
