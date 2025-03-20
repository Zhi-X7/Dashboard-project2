import dash
import joblib
from dash import dcc, html, dash_table
import dash.dependencies as dd
import plotly.graph_objects as go
import pandas as pd
import pickle
import numpy as np
import os

# 1. External stylesheets - using Normalize and Bootstrap for consistent styling
external_stylesheets = [
    'https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css',
    'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'
]

# 2. Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# 3. Load CSV data files
df_raw = pd.read_csv("SouthTower_test_2019_2.csv")
df_real = pd.read_csv("real_results_2019.csv")

# 4. Standardize the Date_Time column format
df_raw['Date_Time'] = pd.to_datetime(df_raw['Date_Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
df_real['Date_Time'] = pd.to_datetime(df_real['Date_Time']).dt.strftime('%Y-%m-%d %H:%M:%S')

# 5. Load pre-trained models
with open("LR_model.sav", 'rb') as f:
    lr_model = pickle.load(f)
with open("RF_model.sav", 'rb') as f:
    rf_model = joblib.load(f)

# 6. Define the feature list for model input
features = ['Power-1', 'Temperature(°C)', 'Solar Radiation(W/m²)', 'Holiday', 'hour', 'sin_hour']

# 7. Prepare input features DataFrame
X_input = df_raw[features]

# 8. Make predictions using both models and round to 3 decimal places
predictions_lr = lr_model.predict(X_input).round(3)
# Convert DataFrame to numpy array for Random Forest prediction to avoid warnings
predictions_rf = rf_model.predict(X_input.values).round(3)

# 9. Build a DataFrame for predictions and set Date_Time as index
predictions_df = pd.DataFrame({
    'Date_Time': df_raw['Date_Time'],
    'LR_Prediction': predictions_lr,
    'RF_Prediction': predictions_rf
})
predictions_df.set_index('Date_Time', inplace=True)

# 10. Save predictions to CSV file
predictions_df.to_csv("model_predictions.csv")

# 11. Define the main layout for the dashboard
app.layout = html.Div(
    style={
        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
        'backgroundColor': '#f5f5f7',
        'color': '#1d1d1f',
        'margin': '0',
        'padding': '0'
    },
    children=[
        # Top header section with background image, title, and date range picker
        html.Div(
            children=[
                # Dashboard title
                html.H1(
                    "IST SouthTower Energy Forecasting Dashboard",
                    style={
                        'fontSize': '2.2rem',
                        'margin': '0',
                        'color': '#fff',
                        'textShadow': '1px 1px 2px rgba(0,0,0,0.6)'
                    }
                ),
                # Date range picker for calendar selection
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
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'marginTop': '20px'
                })
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

        # Tabs for navigation between different sections
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

        # Container for the content corresponding to the selected tab
        html.Div(
            id='tabs-content',
            style={
                'padding': '20px',
                'maxWidth': '1200px',
                'margin': '0 auto'
            }
        )
    ]
)

# Callback to render content for each tab based on selected value and date range
@app.callback(
    dd.Output('tabs-content', 'children'),
    [dd.Input('tabs', 'value'),
     dd.Input('date-picker-range', 'start_date'),
     dd.Input('date-picker-range', 'end_date')]
)
def render_content(tab, start_date, end_date):
    if tab == 'tab1':
        # Tab 1: Raw Data Layout
        return html.Div(
            style={
                'backgroundColor': '#ffffff',
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            },
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
                html.Button(
                    "Update Raw Graph",
                    id='update-raw',
                    n_clicks=0,
                    style={
                        'margin': '10px',
                        'padding': '10px 20px',
                        'fontSize': '1rem',
                        'cursor': 'pointer'
                    }
                ),
                dcc.Graph(id='raw-graph'),
                dash_table.DataTable(
                    id='raw-data-table',
                    # Create table columns based on df_raw columns
                    columns=[{"name": col, "id": col} for col in df_raw.columns],
                    data=[],
                    page_size=10,
                    style_header={
                        'fontWeight': 'bold',
                        'textAlign': 'left',
                        'backgroundColor': '#f0f0f5',
                        'border': 'none'
                    },
                    style_cell={
                        'textAlign': 'left',
                        'padding': '8px',
                        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
                    },
                    style_table={'overflowX': 'auto'}
                )
            ]
        )

    elif tab == 'tab2':
        # Tab 2: Forecast & Metrics Layout
        return html.Div(
            style={
                'backgroundColor': '#ffffff',
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            },
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
        # Tab 3: User Input Forecast Layout (Beautified with Bootstrap)
        return html.Div(
            className="card",
            style={
                'backgroundColor': '#ffffff',
                'padding': '0',
                'borderRadius': '8px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.15)',
                'margin': '20px auto',
                'maxWidth': '800px'
            },
            children=[
                # Card header for title with updated background color to match the Raw Data graph
                html.Div(
                    "Forecasting for Today Based on User Inputs",
                    className="card-header",
                    style={
                        'fontSize': '1.5rem',
                        'fontWeight': 'bold',
                        'backgroundColor': '#e6f7ff',  # Changed from deep blue to light blue
                        'color': '#1d1d1f',          # Dark text for better readability
                        'borderRadius': '8px 8px 0 0'
                    }
                ),
                # Card body for user inputs and button
                html.Div(
                    className="card-body",
                    children=[
                        # First row: Power-1 and Temperature inputs
                        html.Div(
                            className="form-row",
                            children=[
                                html.Div(
                                    className="form-group col-md-6",
                                    children=[
                                        html.Label("Power-1 (kW):"),
                                        dcc.Input(id='power-input', type='number', value=0, className='form-control')
                                    ]
                                ),
                                html.Div(
                                    className="form-group col-md-6",
                                    children=[
                                        html.Label("Temperature (°C):"),
                                        dcc.Input(id='temperature-input', type='number', value=25, className='form-control')
                                    ]
                                )
                            ]
                        ),
                        # Second row: Solar Radiation and Holiday inputs
                        html.Div(
                            className="form-row",
                            children=[
                                html.Div(
                                    className="form-group col-md-6",
                                    children=[
                                        html.Label("Solar Radiation (W/m²):"),
                                        dcc.Input(id='solar-input', type='number', value=500, className='form-control')
                                    ]
                                ),
                                html.Div(
                                    className="form-group col-md-6",
                                    children=[
                                        html.Label("Holiday (0 or 1):"),
                                        dcc.Input(id='holiday-input', type='number', value=0, className='form-control')
                                    ]
                                )
                            ]
                        ),
                        # Third row: Hour and sin_hour inputs
                        html.Div(
                            className="form-row",
                            children=[
                                html.Div(
                                    className="form-group col-md-6",
                                    children=[
                                        html.Label("Hour (0-23):"),
                                        dcc.Input(id='hour-input', type='number', value=12, className='form-control')
                                    ]
                                ),
                                html.Div(
                                    className="form-group col-md-6",
                                    children=[
                                        html.Label("sin_hour:"),
                                        dcc.Input(id='sin-hour-input', type='number', value=0, className='form-control')
                                    ]
                                )
                            ]
                        ),
                        # Predict button spanning full width with modified style for浅蓝色
                        html.Button(
                            "Predict",
                            id='predict-button',
                            className="btn mt-3",  # 移除了 btn-primary 以防止覆盖自定义样式
                            style={
                                'width': '100%',
                                'backgroundColor': '#ADD8E6',  # 浅蓝色背景
                                'border': 'none',
                                'color': '#1d1d1f'
                            }
                        ),
                        # Area to display prediction results
                        html.Div(
                            id='prediction-output',
                            style={'marginTop': '20px', 'fontWeight': 'bold', 'textAlign': 'center'}
                        )
                    ]
                )
            ]
        )

    elif tab == 'tab4':
        # Tab 4: Exploratory Data Analysis (EDA) Layout
        return html.Div(
            style={
                'backgroundColor': '#ffffff',
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            },
            children=[
                html.H3("Exploratory Data Analysis"),
                html.Div([
                    html.Label("Select Graph Type:"),
                    dcc.Dropdown(
                        id='eda-graph-type',
                        options=[
                            {'label': 'Line Chart', 'value': 'line'},
                            {'label': 'Scatter Plot', 'value': 'scatter'},
                            {'label': 'Histogram', 'value': 'histogram'}
                        ],
                        value='line',
                        clearable=False
                    )
                ], style={'margin': '10px', 'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.Label("Select Variables:"),
                    dcc.Dropdown(
                        id='eda-variables',
                        options=[{'label': col, 'value': col} for col in df_raw.columns if col != 'Date_Time'],
                        value=[features[0]],
                        multi=True
                    )
                ], style={'margin': '10px', 'width': '48%', 'display': 'inline-block'}),
                html.Button(
                    "Update Graph",
                    id='update-eda',
                    n_clicks=0,
                    style={'margin': '10px', 'padding': '10px 20px', 'fontSize': '1rem', 'cursor': 'pointer'}
                ),
                dcc.Graph(id='eda-graph')
            ]
        )

    elif tab == 'tab5':
        # Tab 5: Feature Selection Layout
        return html.Div(
            style={
                'backgroundColor': '#ffffff',
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            },
            children=[
                html.H3("Feature Selection"),
                html.Label("Select Features:"),
                dcc.Dropdown(
                    id='fs-features',
                    options=[{'label': col, 'value': col} for col in features],
                    value=features,
                    multi=True
                ),
                html.Label("Select Feature Selection Method:"),
                dcc.Dropdown(
                    id='fs-method',
                    options=[
                        {'label': 'Correlation Analysis', 'value': 'correlation'},
                        {'label': 'Recursive Feature Elimination', 'value': 'rfe'}
                    ],
                    value='correlation',
                    clearable=False
                ),
                html.Button(
                    "Run Feature Selection",
                    id='run-fs',
                    n_clicks=0,
                    style={'marginTop': '10px', 'padding': '10px 20px', 'fontSize': '1rem', 'cursor': 'pointer'}
                ),
                html.Div(id='fs-output', style={'marginTop': '20px', 'fontWeight': 'bold'})
            ]
        )

# Callback to update the Raw Data graph and table (Tab 1)
@app.callback(
    [
        dd.Output('raw-graph', 'figure'),
        dd.Output('raw-data-table', 'data')
    ],
    [
        dd.Input('update-raw', 'n_clicks'),
        dd.Input('date-picker-range', 'start_date'),
        dd.Input('date-picker-range', 'end_date')
    ],
    [dd.State('raw-graph-type', 'value')]
)
def update_raw_graph_table(n_clicks, start_date, end_date, raw_graph_type):
    """
    Update the Raw Data graph and table based on the selected graph type and date range.
    Also adds reference lines (mean values) to the graph.
    """
    # 1. Filter data by date range
    filtered_raw = df_raw[(df_raw['Date_Time'] >= start_date) & (df_raw['Date_Time'] <= end_date)]

    # 2. Create the graph figure
    fig = go.Figure()
    if raw_graph_type == 'line':
        for feat in features:
            fig.add_trace(go.Scatter(
                x=filtered_raw['Date_Time'],
                y=filtered_raw[feat].round(3),
                mode='lines',
                name=feat
            ))
        # Add horizontal mean lines for each feature
        for feat in features:
            mean_val = filtered_raw[feat].mean()
            fig.add_hline(
                y=mean_val,
                line_dash='dot',
                line_color='cyan',
                annotation_text=f"{feat} mean: {mean_val:.2f}",
                annotation_position="top right",
                annotation_font_color='cyan'
            )
    elif raw_graph_type == 'scatter':
        for feat in features:
            fig.add_trace(go.Scatter(
                x=filtered_raw['Date_Time'],
                y=filtered_raw[feat].round(3),
                mode='markers',
                name=feat
            ))
        # Add horizontal mean lines for each feature
        for feat in features:
            mean_val = filtered_raw[feat].mean()
            fig.add_hline(
                y=mean_val,
                line_dash='dot',
                line_color='cyan',
                annotation_text=f"{feat} mean: {mean_val:.2f}",
                annotation_position="top right",
                annotation_font_color='cyan'
            )
    elif raw_graph_type == 'histogram':
        for feat in features:
            fig.add_trace(go.Histogram(
                x=filtered_raw[feat],
                name=feat,
                opacity=0.75
            ))
        fig.update_layout(barmode='overlay')
        # Add vertical mean lines for each feature
        for feat in features:
            mean_val = filtered_raw[feat].mean()
            fig.add_vline(
                x=mean_val,
                line_dash='dot',
                line_color='cyan',
                annotation_text=f"{feat} mean: {mean_val:.2f}",
                annotation_position="top right",
                annotation_font_color='cyan'
            )

    # Update graph layout with a light blue background
    fig.update_layout(
        title="Raw Data with Reference Lines",
        paper_bgcolor='#e6f7ff',
        plot_bgcolor='#e6f7ff',
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
            color='#1d1d1f'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey'
        )
    )

    # 3. Update table data by converting the filtered data to records
    table_data = filtered_raw.round(3).to_dict('records')
    return fig, table_data

# Callback to update Forecast & Metrics (Tab 2)
@app.callback(
    dd.Output('forecast-metrics-container', 'children'),
    [
        dd.Input('forecast-method', 'value'),
        dd.Input('metrics-select', 'value'),
        dd.Input('date-picker-range', 'start_date'),
        dd.Input('date-picker-range', 'end_date')
    ]
)
def update_forecast_metrics(forecast_method, selected_metrics, start_date, end_date):
    # 1. Filter the predictions and actual data based on the date range
    filtered_pred = predictions_df.loc[start_date:end_date]
    filtered_real = df_real[(df_real['Date_Time'] >= start_date) & (df_real['Date_Time'] <= end_date)]

    # 2. Build the forecast graph comparing actual data with predictions
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=filtered_real['Date_Time'],
        y=filtered_real['Power(kW)'],
        mode='lines',
        name='Actual Data'
    ))
    if forecast_method in ['LR', 'both']:
        fig_forecast.add_trace(go.Scatter(
            x=filtered_pred.index,
            y=filtered_pred['LR_Prediction'],
            mode='lines',
            name='Linear Regression'
        ))
    if forecast_method in ['RF', 'both']:
        fig_forecast.add_trace(go.Scatter(
            x=filtered_pred.index,
            y=filtered_pred['RF_Prediction'],
            mode='lines',
            name='Random Forest'
        ))
    # Update layout with light blue background
    fig_forecast.update_layout(
        title="Forecast Comparison",
        paper_bgcolor='#e6f7ff',
        plot_bgcolor='#e6f7ff',
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
            color='#1d1d1f'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey'
        )
    )

    # 3. Build error metrics table (dummy metrics for demonstration)
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

    # 4. Display only the selected metrics
    cols_to_show = ['Methods'] + selected_metrics
    metrics_df = metrics_df[cols_to_show]

    # 5. Ensure NMBE is displayed with 6 decimal places if selected
    if 'NMBE' in metrics_df.columns:
        metrics_df["NMBE"] = metrics_df["NMBE"].apply(lambda x: format(x, ".6f"))

    # 6. Create a DataTable to display the metrics
    table = dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in metrics_df.columns],
        data=metrics_df.to_dict('records'),
        style_header={
            'fontWeight': 'bold',
            'textAlign': 'left',
            'backgroundColor': '#f0f0f5',
            'border': 'none'
        },
        style_cell={
            'textAlign': 'left',
            'padding': '8px',
            'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
        },
        style_table={'overflowX': 'auto'}
    )

    return html.Div(
        style={
            'backgroundColor': '#ffffff',
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        },
        children=[
            dcc.Graph(figure=fig_forecast),
            table
        ]
    )

# Callback to update the Exploratory Data Analysis (EDA) Graph (Tab 4)
@app.callback(
    dd.Output('eda-graph', 'figure'),
    [dd.Input('update-eda', 'n_clicks')],
    [
        dd.State('eda-graph-type', 'value'),
        dd.State('eda-variables', 'value'),
        dd.State('date-picker-range', 'start_date'),
        dd.State('date-picker-range', 'end_date')
    ]
)
def update_eda_graph(n_clicks, graph_type, variables, start_date, end_date):
    # 1. Filter data for the selected date range
    filtered_data = df_raw[(df_raw['Date_Time'] >= start_date) & (df_raw['Date_Time'] <= end_date)]
    fig = go.Figure()

    # 2. Build the graph based on the selected graph type
    if graph_type == 'line':
        for var in variables:
            fig.add_trace(go.Scatter(
                x=filtered_data['Date_Time'],
                y=filtered_data[var],
                mode='lines',
                name=var
            ))
    elif graph_type == 'scatter':
        for var in variables:
            fig.add_trace(go.Scatter(
                x=filtered_data['Date_Time'],
                y=filtered_data[var],
                mode='markers',
                name=var
            ))
    elif graph_type == 'histogram':
        for var in variables:
            fig.add_trace(go.Histogram(
                x=filtered_data[var],
                name=var,
                opacity=0.75
            ))
        fig.update_layout(barmode='overlay')

    # Update graph layout with light blue background
    fig.update_layout(
        title="Exploratory Data Analysis",
        paper_bgcolor='#e6f7ff',
        plot_bgcolor='#e6f7ff',
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
            color='#1d1d1f'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey'
        )
    )
    return fig

# Callback for the User Input Forecast (Tab 3)
@app.callback(
    dd.Output('prediction-output', 'children'),
    [dd.Input('predict-button', 'n_clicks')],
    [
        dd.State('power-input', 'value'),
        dd.State('temperature-input', 'value'),
        dd.State('solar-input', 'value'),
        dd.State('holiday-input', 'value'),
        dd.State('hour-input', 'value'),
        dd.State('sin-hour-input', 'value')
    ]
)
def predict_power(n_clicks, power_1, temperature, solar, holiday, hour, sin_hour):
    # 1. Check if the predict button has been clicked
    if not n_clicks:
        return ""
    # 2. Build a single row input DataFrame for prediction
    input_data = pd.DataFrame([[
        power_1, temperature, solar, holiday, hour, sin_hour
    ]], columns=['Power-1', 'Temperature(°C)', 'Solar Radiation(W/m²)', 'Holiday', 'hour', 'sin_hour'])

    # 3. Make predictions using both models
    lr_pred = lr_model.predict(input_data)[0]
    rf_pred = rf_model.predict(input_data.values)[0]

    # 4. Return the predictions formatted to three decimal places
    return (
        f"Linear Regression Prediction: {lr_pred:.3f} kW  |  "
        f"Random Forest Prediction: {rf_pred:.3f} kW"
    )

# Callback for Feature Selection (Tab 5)
@app.callback(
    dd.Output('fs-output', 'children'),
    [dd.Input('run-fs', 'n_clicks')],
    [dd.State('fs-features', 'value'),
     dd.State('fs-method', 'value')]
)
def run_feature_selection(n_clicks, selected_features, method):
    # 1. Check if the feature selection button has been clicked
    if not n_clicks:
        return ""
    # 2. Dummy simulation: select half of the features as a dummy result
    dummy_selected = selected_features[:max(1, len(selected_features) // 2)]
    result = (
        f"Feature selection using {method} on features: {', '.join(selected_features)}. "
        f"(Dummy result: Selected features: {', '.join(dummy_selected)})"
    )
    return html.Div(result, style={'marginTop': '20px', 'fontWeight': 'bold'})

# Configure the server for deployment
server = app.server

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
