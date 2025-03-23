import dash
import joblib
from dash import dcc, html, dash_table
import dash.dependencies as dd
import plotly.graph_objects as go
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# 1. Set external stylesheets for consistent styling
external_stylesheets = [
    'https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css',
    'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'
]

# 2. Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# 3. Load CSV data files for different pages
df_features = pd.read_csv("SouthTower_test_2019_2.csv")
df_raw = pd.read_csv("SouthTower_test_2019_prepared.csv")
df_real = pd.read_csv("real_results_2019.csv")

# 4. Standardize the Date_Time column format in all datasets
df_features['Date_Time'] = pd.to_datetime(df_features['Date_Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
df_raw['Date_Time'] = pd.to_datetime(df_raw['Date_Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
df_real['Date_Time'] = pd.to_datetime(df_real['Date_Time']).dt.strftime('%Y-%m-%d %H:%M:%S')

# 5. Load pre-trained models
with open("LR_model.sav", 'rb') as f:
    lr_model = pickle.load(f)
with open("RF_model.sav", 'rb') as f:
    rf_model = joblib.load(f)

# 6. Define the list of features for model prediction
features = ['Power-1', 'Temperature(°C)', 'Solar Radiation(W/m²)', 'Holiday', 'hour', 'sin_hour']

# 7. Prepare input features and generate predictions using both models
X_input = df_features[features]
predictions_lr = lr_model.predict(X_input).round(3)
predictions_rf = rf_model.predict(X_input.values).round(3)

# 8. Create a DataFrame for predictions and save it to CSV
predictions_df = pd.DataFrame({
    'Date_Time': df_features['Date_Time'],
    'LR_Prediction': predictions_lr,
    'RF_Prediction': predictions_rf
})
predictions_df.set_index('Date_Time', inplace=True)
predictions_df.to_csv("model_predictions.csv")

# 9. Define the main layout for the dashboard
app.layout = html.Div(
    style={
        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
        'backgroundColor': '#f5f5f7',
        'color': '#1d1d1f',
        'margin': '0',
        'padding': '0'
    },
    children=[
        html.Div(
            children=[
                html.H1(
                    "IST SouthTower Energy Forecasting Dashboard",
                    style={
                        'fontSize': '2.2rem',
                        'margin': '0',
                        'color': '#fff',
                        'textShadow': '1px 1px 2px rgba(0,0,0,0.6)'
                    }
                ),
                html.Div([
                    html.Label("Calendar:", style={
                        'fontWeight': 'bold',
                        'marginRight': '10px',
                        'color': '#fff',
                        'textShadow': '1px 1px 2px rgba(0,0,0,0.6)'
                    }),
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        start_date="2019-01-01 00:00:00",
                        end_date="2019-03-31 23:59:59",
                        display_format='YYYY-MM-DD',
                        style={
                            'marginBottom': '20px',
                            'backgroundColor': 'rgba(255,255,255,0.7)',
                            'border': 'none',
                            'borderRadius': '4px',
                            'padding': '4px'
                        }
                    )
                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginTop': '20px'})
            ],
            style={
                'backgroundImage': 'url("/assets/ist_logo.png")',
                'backgroundSize': 'cover',
                'backgroundRepeat': 'no-repeat',
                'backgroundPosition': 'center',
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'center',
                'justifyContent': 'center',
                'padding': '40px 20px',
                'textAlign': 'center',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            }
        ),
        dcc.Tabs(
            id="tabs",
            value='tab1',
            children=[
                dcc.Tab(label='Raw Data', value='tab1', style={'padding': '10px'}),
                dcc.Tab(label='Forecast & Metrics', value='tab2', style={'padding': '10px'}),
                dcc.Tab(label='User Input Forecast', value='tab3', style={'padding': '10px'}),
                dcc.Tab(label='Exploratory Data Analysis', value='tab4', style={'padding': '10px'}),
                dcc.Tab(label='Feature Selection', value='tab5', style={'padding': '10px'})
            ],
            style={
                'fontSize': '1.1rem',
                'fontWeight': 'bold',
                'margin': '20px auto',
                'maxWidth': '1200px'
            }
        ),
        html.Div(
            id='tabs-content',
            style={'padding': '20px', 'maxWidth': '1200px', 'margin': '0 auto'}
        )
    ]
)

# 10. Render page content based on selected tab and date range
@app.callback(
    dd.Output('tabs-content', 'children'),
    [dd.Input('tabs', 'value'),
     dd.Input('date-picker-range', 'start_date'),
     dd.Input('date-picker-range', 'end_date')]
)
def render_content(tab, start_date, end_date):
    if tab == 'tab1':
        return html.Div(
            style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '8px',
                   'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'},
            children=[
                html.Div([
                    html.Label("Select Graph Type (Raw):"),
                    dcc.Dropdown(
                        id='raw-graph-type',
                        options=[
                            {'label': 'Line Chart', 'value': 'line'},
                            {'label': 'Scatter Plot', 'value': 'scatter'},
                            {'label': 'Histogram', 'value': 'histogram'}
                        ],
                        value='line',
                        clearable=False
                    )
                ], style={'margin': '10px', 'width': '30%'}),
                html.Button("Update Raw Graph", id='update-raw', n_clicks=0,
                            style={'margin': '10px', 'padding': '10px 20px', 'fontSize': '1rem', 'cursor': 'pointer'}),
                dcc.Graph(id='raw-graph'),
                dash_table.DataTable(
                    id='raw-data-table',
                    columns=[{"name": col, "id": col} for col in df_raw.columns],
                    data=[],
                    page_size=10,
                    style_header={'fontWeight': 'bold', 'textAlign': 'left', 'backgroundColor': '#f0f0f5',
                                  'border': 'none'},
                    style_cell={'textAlign': 'left', 'padding': '8px',
                                'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'},
                    style_table={'overflowX': 'auto'}
                )
            ]
        )
    elif tab == 'tab2':
        return html.Div(
            style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '8px',
                   'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'},
            children=[
                html.Div([
                    html.Label("Select Forecasting Method:"),
                    dcc.Dropdown(
                        id='forecast-method',
                        options=[
                            {'label': 'Linear Regression', 'value': 'LR'},
                            {'label': 'Random Forest', 'value': 'RF'},
                            {'label': 'Both', 'value': 'both'}
                        ],
                        value='both',
                        clearable=False
                    ),
                    html.Label("Select Metrics to Display:"),
                    dcc.Dropdown(
                        id='metrics-select',
                        options=[
                            {'label': 'MAE', 'value': 'MAE'},
                            {'label': 'MBE', 'value': 'MBE'},
                            {'label': 'MSE', 'value': 'MSE'},
                            {'label': 'RMSE', 'value': 'RMSE'},
                            {'label': 'cvMSE', 'value': 'cvMSE'},
                            {'label': 'NMBE', 'value': 'NMBE'}
                        ],
                        value=['MAE', 'MBE', 'MSE', 'RMSE', 'cvMSE', 'NMBE'],
                        multi=True
                    )
                ], style={'margin': '20px'}),
                html.Div(id='forecast-metrics-container')
            ]
        )
    elif tab == 'tab3':
        return html.Div(
            className="card",
            style={'backgroundColor': '#ffffff', 'padding': '0', 'borderRadius': '8px',
                   'boxShadow': '0 2px 8px rgba(0,0,0,0.15)', 'margin': '20px auto', 'maxWidth': '800px'},
            children=[
                html.Div("Forecasting for Today Based on User Inputs",
                         className="card-header",
                         style={'fontSize': '1.5rem', 'fontWeight': 'bold', 'backgroundColor': '#e6f7ff',
                                'color': '#1d1d1f', 'borderRadius': '8px 8px 0 0'}),
                html.Div(
                    className="card-body",
                    children=[
                        html.Div(
                            className="form-row",
                            children=[
                                html.Div(className="form-group col-md-6",
                                         children=[html.Label("Power-1 (kW):"),
                                                   dcc.Input(id='power-input', type='number', value=0,
                                                             className='form-control')]),
                                html.Div(className="form-group col-md-6",
                                         children=[html.Label("Temperature (°C):"),
                                                   dcc.Input(id='temperature-input', type='number', value=25,
                                                             className='form-control')])
                            ]
                        ),
                        html.Div(
                            className="form-row",
                            children=[
                                html.Div(className="form-group col-md-6",
                                         children=[html.Label("Solar Radiation (W/m²):"),
                                                   dcc.Input(id='solar-input', type='number', value=500,
                                                             className='form-control')]),
                                html.Div(className="form-group col-md-6",
                                         children=[html.Label("Holiday (0 or 1):"),
                                                   dcc.Input(id='holiday-input', type='number', value=0,
                                                             className='form-control')])
                            ]
                        ),
                        html.Div(
                            className="form-row",
                            children=[
                                html.Div(className="form-group col-md-6",
                                         children=[html.Label("Hour (0-23):"),
                                                   dcc.Input(id='hour-input', type='number', value=12,
                                                             className='form-control')]),
                                html.Div(className="form-group col-md-6",
                                         children=[html.Label("sin_hour:"),
                                                   dcc.Input(id='sin-hour-input', type='number', value=0,
                                                             className='form-control')])
                            ]
                        ),
                        html.Button("Predict", id='predict-button',
                                    style={'margin': '10px', 'padding': '10px 20px', 'fontSize': '1rem', 'cursor': 'pointer'}),
                        html.Div(id='prediction-output',
                                 style={'marginTop': '20px', 'fontWeight': 'bold', 'textAlign': 'center'})
                    ]
                )
            ]
        )
    elif tab == 'tab4':
        return html.Div(
            style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '8px',
                   'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'},
            children=[
                html.H3("Exploratory Data Analysis"),
                html.Div([
                    html.Label("Select Graph Type:"),
                    dcc.Dropdown(
                        id='eda-graph-type',
                        options=[
                            {'label': 'Line Chart', 'value': 'line'},
                            {'label': 'Scatter Plot', 'value': 'scatter'},
                            {'label': 'Histogram', 'value': 'histogram'},
                            {'label': 'Box Plot', 'value': 'boxplot'}
                        ],
                        value='line',
                        clearable=False
                    )
                ], style={'margin': '10px', 'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.Label("Select Variables:"),
                    dcc.Dropdown(
                        id='eda-variables',
                        options=[{'label': col, 'value': col} for col in df_features.columns if col != 'Date_Time'],
                        value=[features[0]],
                        multi=True
                    )
                ], style={'margin': '10px', 'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.Label("Select Outliers:"),
                    dcc.Dropdown(
                        id='outliers-select',
                        options=[
                            {'label': "Tukey's method", 'value': 'tukey'},
                            {'label': 'No Outlier', 'value': 'no_outlier'}
                        ],
                        value='tukey',
                        clearable=False
                    )
                ], style={'margin': '10px', 'width': '48%', 'display': 'inline-block'}),
                html.Div(
                    html.Button("Update Graph", id='update-eda', n_clicks=0,
                                style={'padding': '10px 20px', 'fontSize': '1rem', 'cursor': 'pointer'}),
                    style={'margin': '20px 10px 10px 10px', 'textAlign': 'left'}
                ),
                dcc.Graph(id='eda-graph'),
                dash_table.DataTable(
                    id='eda-stats-table',
                    columns=[],
                    data=[],
                    style_header={'fontWeight': 'bold', 'textAlign': 'left', 'backgroundColor': '#f0f0f5', 'border': 'none'},
                    style_cell={'textAlign': 'left', 'padding': '8px',
                                'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'},
                    style_table={'overflowX': 'auto', 'marginTop': '20px'}
                )
            ]
        )
    elif tab == 'tab5':
        return html.Div(
            style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '8px',
                   'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'},
            children=[
                html.H3("Feature Selection"),
                html.Label("Select Features:"),
                dcc.Dropdown(
                    id='fs-features',
                    options=[{'label': col, 'value': col} for col in df_raw.columns if col != 'Date_Time'],
                    value=[col for col in df_raw.columns if col != 'Date_Time'],
                    multi=True
                ),
                html.Label("Select Feature Selection Method:"),
                dcc.Dropdown(
                    id='fs-method',
                    options=[
                        {'label': 'Filter Methods (KBest)', 'value': 'kbest'},
                        {'label': 'Wrapper Methods (RFE)', 'value': 'rfe'},
                        {'label': 'Ensemble Methods', 'value': 'ensemble'}
                    ],
                    value='kbest',
                    clearable=False
                ),
                html.Button("Run Feature Selection", id='run-fs', n_clicks=0,
                            style={'marginTop': '10px', 'padding': '10px 20px', 'fontSize': '1rem', 'cursor': 'pointer'}),
                html.Div(id='fs-output', style={'marginTop': '20px', 'fontWeight': 'bold'})
            ]
        )

# 11. Update the Raw Data graph and table based on selected date range and graph type
@app.callback(
    [dd.Output('raw-graph', 'figure'),
     dd.Output('raw-data-table', 'data')],
    [dd.Input('update-raw', 'n_clicks'),
     dd.Input('date-picker-range', 'start_date'),
     dd.Input('date-picker-range', 'end_date')],
    [dd.State('raw-graph-type', 'value')]
)
def update_raw_graph_table(n_clicks, start_date, end_date, raw_graph_type):
    filtered_data = df_raw[(df_raw['Date_Time'] >= start_date) & (df_raw['Date_Time'] <= end_date)]
    fig = go.Figure()
    if raw_graph_type == 'line':
        for feat in filtered_data.columns:
            if feat == 'Date_Time': continue
            fig.add_trace(go.Scatter(x=filtered_data['Date_Time'],
                                     y=filtered_data[feat].round(3),
                                     mode='lines',
                                     name=feat))
    elif raw_graph_type == 'scatter':
        for feat in filtered_data.columns:
            if feat == 'Date_Time': continue
            fig.add_trace(go.Scatter(x=filtered_data['Date_Time'],
                                     y=filtered_data[feat].round(3),
                                     mode='markers',
                                     name=feat))
    elif raw_graph_type == 'histogram':
        for feat in filtered_data.columns:
            if feat == 'Date_Time': continue
            fig.add_trace(go.Histogram(x=filtered_data[feat],
                                       name=feat,
                                       opacity=0.75))
        fig.update_layout(barmode='overlay')
    fig.update_layout(
        title="Raw Data",
        paper_bgcolor='#e6f7ff',
        plot_bgcolor='#e6f7ff',
        font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
                  color='#1d1d1f'),
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey')
    )
    table_data = filtered_data.round(3).to_dict('records')
    return fig, table_data

# 12. Update the Forecast & Metrics section based on selected method and date range
@app.callback(
    dd.Output('forecast-metrics-container', 'children'),
    [dd.Input('forecast-method', 'value'),
     dd.Input('metrics-select', 'value'),
     dd.Input('date-picker-range', 'start_date'),
     dd.Input('date-picker-range', 'end_date')]
)
def update_forecast_metrics(forecast_method, selected_metrics, start_date, end_date):
    filtered_pred = predictions_df.loc[start_date:end_date]
    filtered_real = df_real[(df_real['Date_Time'] >= start_date) & (df_real['Date_Time'] <= end_date)]
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=filtered_real['Date_Time'],
                                      y=filtered_real['Power(kW)'],
                                      mode='lines',
                                      name='Actual Data'))
    if forecast_method in ['LR', 'both']:
        fig_forecast.add_trace(go.Scatter(x=filtered_pred.index,
                                          y=filtered_pred['LR_Prediction'],
                                          mode='lines',
                                          name='Linear Regression'))
    if forecast_method in ['RF', 'both']:
        fig_forecast.add_trace(go.Scatter(x=filtered_pred.index,
                                          y=filtered_pred['RF_Prediction'],
                                          mode='lines',
                                          name='Random Forest'))
    fig_forecast.update_layout(
        title="Forecast Comparison",
        paper_bgcolor='#e6f7ff',
        plot_bgcolor='#e6f7ff',
        font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
                  color='#1d1d1f'),
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey')
    )
    metrics_data = {
        "Methods": ["Linear Regression", "Random Forest"],
        "MAE": [111.637, 44.494],
        "MBE": [-0.192, 13.352],
        "MSE": [23177.904, 5885.171],
        "RMSE": [152.243, 76.715],
        "cvMSE": [0.121, 0.061],
        "NMBE": [-0.000123, 0.107456]
    }
    metrics_df = pd.DataFrame(metrics_data).round(6)
    cols_to_show = ['Methods'] + selected_metrics
    metrics_df = metrics_df[cols_to_show]
    if 'NMBE' in metrics_df.columns:
        metrics_df["NMBE"] = metrics_df["NMBE"].apply(lambda x: format(x, ".6f"))
    table = dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in metrics_df.columns],
        data=metrics_df.to_dict('records'),
        style_header={'fontWeight': 'bold', 'textAlign': 'left', 'backgroundColor': '#f0f0f5', 'border': 'none'},
        style_cell={'textAlign': 'left', 'padding': '8px',
                    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'},
        style_table={'overflowX': 'auto'}
    )
    return html.Div(style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '8px',
                           'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'},
                    children=[dcc.Graph(figure=fig_forecast), table])

# 13. Update the EDA graph based on selected options
@app.callback(
    dd.Output('eda-graph', 'figure'),
    [dd.Input('update-eda', 'n_clicks')],
    [dd.State('eda-graph-type', 'value'),
     dd.State('eda-variables', 'value'),
     dd.State('date-picker-range', 'start_date'),
     dd.State('date-picker-range', 'end_date'),
     dd.State('outliers-select', 'value')]
)
def update_eda_graph(n_clicks, graph_type, variables, start_date, end_date, outliers_method):
    filtered_data = df_features[(df_features['Date_Time'] >= start_date) & (df_features['Date_Time'] <= end_date)]
    fig = go.Figure()
    if graph_type == 'line':
        for var in variables:
            fig.add_trace(go.Scatter(x=filtered_data['Date_Time'],
                                     y=filtered_data[var],
                                     mode='lines',
                                     name=var))
    elif graph_type == 'scatter':
        for var in variables:
            fig.add_trace(go.Scatter(x=filtered_data['Date_Time'],
                                     y=filtered_data[var],
                                     mode='markers',
                                     name=var))
    elif graph_type == 'histogram':
        for var in variables:
            fig.add_trace(go.Histogram(x=filtered_data[var],
                                       name=var,
                                       opacity=0.75))
        fig.update_layout(barmode='overlay')
    elif graph_type == 'boxplot':
        for var in variables:
            fig.add_trace(go.Box(y=filtered_data[var], name=var))
    if outliers_method == 'tukey' and graph_type in ['line', 'scatter', 'histogram']:
        for var in variables:
            mean_val = filtered_data[var].mean()
            q1 = filtered_data[var].quantile(0.25)
            q3 = filtered_data[var].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            if graph_type in ['line', 'scatter']:
                fig.add_hline(y=mean_val, line_dash="dash", line_color="red", annotation_text=f"{var} mean",
                              annotation_font_color="red")
                fig.add_hline(y=lower_bound, line_dash="dot", line_color="green", annotation_text=f"{var} lower",
                              annotation_font_color="green")
                fig.add_hline(y=upper_bound, line_dash="dot", line_color="green", annotation_text=f"{var} upper",
                              annotation_font_color="green")
            elif graph_type == 'histogram':
                fig.add_vline(x=mean_val, line_dash="dash", line_color="red", annotation_text=f"{var} mean",
                              annotation_font_color="red")
                fig.add_vline(x=lower_bound, line_dash="dot", line_color="green", annotation_text=f"{var} lower",
                              annotation_font_color="green")
                fig.add_vline(x=upper_bound, line_dash="dot", line_color="green", annotation_text=f"{var} upper",
                              annotation_font_color="green")
    fig.update_layout(
        title="Exploratory Data Analysis",
        paper_bgcolor='#e6f7ff',
        plot_bgcolor='#e6f7ff',
        font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
                  color='#1d1d1f'),
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey')
    )
    return fig

# 14. Update the EDA statistics table with descriptive statistics for selected variables
@app.callback(
    [dd.Output('eda-stats-table', 'data'),
     dd.Output('eda-stats-table', 'columns')],
    [dd.Input('update-eda', 'n_clicks')],
    [dd.State('eda-variables', 'value'),
     dd.State('date-picker-range', 'start_date'),
     dd.State('date-picker-range', 'end_date')]
)
def update_eda_stats(n_clicks, variables, start_date, end_date):
    filtered_data = df_features[(df_features['Date_Time'] >= start_date) & (df_features['Date_Time'] <= end_date)]
    stats_list = []
    for var in variables:
        stats_list.append({
            "Variable": var,
            "Mean": round(filtered_data[var].mean(), 3),
            "Std": round(filtered_data[var].std(), 3),
            "Variance": round(filtered_data[var].var(), 3),
            "Min": round(filtered_data[var].min(), 3),
            "Max": round(filtered_data[var].max(), 3)
        })
    columns = [{"name": col, "id": col} for col in ["Variable", "Mean", "Std", "Variance", "Min", "Max"]]
    return stats_list, columns

# 15. Generate predictions based on user input values
@app.callback(
    dd.Output('prediction-output', 'children'),
    [dd.Input('predict-button', 'n_clicks')],
    [dd.State('power-input', 'value'),
     dd.State('temperature-input', 'value'),
     dd.State('solar-input', 'value'),
     dd.State('holiday-input', 'value'),
     dd.State('hour-input', 'value'),
     dd.State('sin-hour-input', 'value')]
)
def predict_power(n_clicks, power_1, temperature, solar, holiday, hour, sin_hour):
    if not n_clicks:
        return ""
    input_data = pd.DataFrame([[power_1, temperature, solar, holiday, hour, sin_hour]],
                              columns=['Power-1', 'Temperature(°C)', 'Solar Radiation(W/m²)', 'Holiday', 'hour', 'sin_hour'])
    lr_pred = lr_model.predict(input_data)[0]
    rf_pred = rf_model.predict(input_data.values)[0]
    return (f"Linear Regression Prediction: {lr_pred:.3f} kW  |  Random Forest Prediction: {rf_pred:.3f} kW")

# 16. Run feature selection using the chosen method on the selected features
@app.callback(
    dd.Output('fs-output', 'children'),
    [dd.Input('run-fs', 'n_clicks')],
    [dd.State('fs-features', 'value'),
     dd.State('fs-method', 'value')]
)
def run_feature_selection(n_clicks, selected_features, method):
    if not n_clicks:
        return ""
    if "Power(kW)" not in df_raw.columns:
        return html.Div("Target variable 'Power(kW)' not found in data.",
                        style={'marginTop': '20px', 'fontWeight': 'bold'})
    candidate_features = [f for f in selected_features if f != "Power(kW)"]
    X = df_raw[candidate_features]
    y = df_raw["Power(kW)"]
    if method == 'kbest':
        selector = SelectKBest(score_func=f_regression, k=len(candidate_features))
        selector.fit(X, y)
        scores = selector.scores_
        features_scores = sorted(zip(candidate_features, scores), key=lambda x: x[1], reverse=True)
        selected = [feat for feat, score in features_scores][:max(1, len(candidate_features) // 2)]
        result = ("Using Filter Method (KBest) [Tukey‘s methods were used in the EDA for outlier thresholds].<br>"
                  "Feature ranking: " + ", ".join([f"{feat} (score: {score:.2f})" for feat, score in features_scores]) +
                  f"<br>Selected features: {', '.join(selected)}.")
    elif method == 'rfe':
        estimator = LinearRegression()
        n_features_to_select = max(1, len(candidate_features) // 2)
        selector = RFE(estimator, n_features_to_select=n_features_to_select)
        selector = selector.fit(X, y)
        ranking = selector.ranking_
        features_ranking = sorted(zip(candidate_features, ranking), key=lambda x: x[1])
        selected = [feat for feat, rank in features_ranking if rank == 1]
        result = ("Using Wrapper Method (RFE).<br>"
                  "Feature ranking: " + ", ".join([f"{feat} (rank: {rank})" for feat, rank in features_ranking]) +
                  f"<br>Selected features: {', '.join(selected)}.")
    elif method == 'ensemble':
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        features_importance = sorted(zip(candidate_features, importances), key=lambda x: x[1], reverse=True)
        selected = [feat for feat, imp in features_importance][:max(1, len(candidate_features) // 2)]
        result = ("Using Ensemble Method (Random Forest).<br>"
                  "Feature importances: " + ", ".join(
            [f"{feat} (importance: {imp:.2f})" for feat, imp in features_importance]) +
                  f"<br>Selected features: {', '.join(selected)}.")
    else:
        result = "No valid feature selection method selected."
    return html.Div(result, style={'marginTop': '20px', 'fontWeight': 'bold'})

# 17. Configure the server for deployment
server = app.server

# 18. Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
