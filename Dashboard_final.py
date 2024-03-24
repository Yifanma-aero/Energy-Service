import dash
from dash import html
import dash_core_components as dcc
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
import pickle
from sklearn import  metrics
import numpy as np

#%% Import and forecast
# Load data
df_test = pd.read_csv('South Tower_2019_test.csv')
df_test['Date'] = pd.to_datetime (df_test['Date']) # create a new column 'data time' of datetime type
df_test2=df_test.iloc[:,2:7]
X2=df_test2.values
X2=X2[:,0:3]

fig1 = px.line(df_test, x="Date", y=df_test.columns[1:3])# Creates a figure with the raw data

df_select = pd.DataFrame({'feature': ['Power-1','Temperature','Hour','Work day','Holiday']})
df_select['kBest'] = [1.24299385, 0.18877153, 0.54106774, 0.17534085, 0.13314675]
df_select['random forest'] = [0.84708821, 0.04769656, 0.08796199, 0.0117631, 0.00549014]

df_real = df_test.iloc[:,0:2]
y2=df_real['South Tower (kWh)'].values    # real power data

#Load and run LR model
with open('LR_model.pkl','rb') as file: LR_model2=pickle.load(file)
y2_pred_LR = LR_model2.predict(X2)
#Evaluate errors
MAE_LR=round(metrics.mean_absolute_error(y2,y2_pred_LR),3)
MBE_LR=round(np.mean(y2-y2_pred_LR),3)
MSE_LR=round(metrics.mean_squared_error(y2,y2_pred_LR),3)
RMSE_LR=round( np.sqrt(metrics.mean_squared_error(y2,y2_pred_LR)),3)
cvRMSE_LR=round(RMSE_LR/np.mean(y2),3)
NMBE_LR=round(MBE_LR/np.mean(y2),3)

#Load NN model
with open('NN_model.pkl','rb') as file: NN_model2=pickle.load(file)
y2_pred_NN = NN_model2.predict(X2)
#Evaluate errors
MAE_NN=round(metrics.mean_absolute_error(y2,y2_pred_NN),3)
MBE_NN=round(np.mean(y2-y2_pred_NN) ,3)
MSE_NN=round(metrics.mean_squared_error(y2,y2_pred_NN),3) 
RMSE_NN= round(np.sqrt(metrics.mean_squared_error(y2,y2_pred_NN)),3)
cvRMSE_NN=round(RMSE_NN/np.mean(y2),3)
NMBE_NN=round(MBE_NN/np.mean(y2),3)

# Create data frames with predictin results and error metrics 
df_metrics = pd.DataFrame({'Error': ['MAE', 'MBE', 'MSE', 'RMSE', 'cvRMSE', 'NMBE']})
df_metrics['LinearRegression'] = [MAE_LR, MBE_LR, MSE_LR, RMSE_LR, cvRMSE_LR, NMBE_LR]
df_metrics['Neural Network'] = [MAE_NN, MBE_NN, MSE_NN, RMSE_NN, cvRMSE_NN, NMBE_NN]

d={'Date':df_real['Date'].values, 'LinearRegression': y2_pred_LR,'Neural Network': y2_pred_NN}
df_forecast=pd.DataFrame(data=d)
# merge real and forecast results and creates a figure with it
df_results=pd.merge(df_real,df_forecast, on='Date')

fig2 = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:4])


#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)

# Sample data for demonstration
app.layout = html.Div([
    html.H1('IST Energy Yearly Consumption'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw data', value='tab-1'),
        dcc.Tab(label='Feature Selection', value='tab-2'),
        dcc.Tab(label='Forecast', value='tab-3'),
    ]),
    html.Div(id='tabs-content'),    
])

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H2('Raw data Content'),
                    dcc.RadioItems(
                        id='radio1',
                        options=[
                            {'label': 'Power-1', 'value': 'Power-1'},
                            {'label': 'temperature (°C)', 'value': 'temp_C'},
                            {'label': 'Hour', 'value': 'Hour'},
                            {'label': 'Work Day', 'value': 'Work Day'},
                            {'label': 'Holiday', 'value': 'Holiday'},
                        ],
                        value='Power-1'
                    ),
            html.Div([
                html.Div([
                    html.H3('Table'),
                    # html.Div(id='table-update1' ),
                    html.Div(id='table-container1' ),
                ], className="six columns"),
                html.Div([
                    html.H3('Graph'),
                    html.Div(id='graph-container1')
                ], className="six columns"),
            ], className="row")
        ])

    if tab == 'tab-2':
        return html.Div([
            html.H2('Feature selection'),
                    dcc.RadioItems(
                        id='radio2',
                        options=[
                            {'label': 'kBest', 'value': 'kBest'},
                            {'label': 'random forest', 'value': 'random forest'}
                        ],
                        value='kBest'
                    ),
            html.Div([
                html.Div([
                    html.H3('Table'),
                    html.Div(id='table-update2' ),
                    html.Div(id='table-container2' ),
                ], className="six columns"),
                html.Div([
                    html.H3('Graph'),

                    html.Div(id='graph-container2')
                ], className="six columns"),
            ], className="row")
        ])

    elif tab == 'tab-3':
        return html.Div([
            html.H2('Forecast'),
                    dcc.RadioItems(
                        id='radio3',
                        options=[
                            {'label': 'Linear Regression', 'value': 'LinearRegression'},
                            {'label': 'Neural Network', 'value': 'Neural Network'}
                        ],
                        value='LinearRegression'
                    ),
            html.Div([
                html.Div([
                    html.H3('Table'),
                    html.Div(id='table-container3' ),
                ], className="six columns"),
                html.Div([
                    html.H3('Graph'),

                    html.Div(id='graph-container3')
                ], className="six columns"),
            ], className="row")
        ])

@app.callback(
    Output('table-container1', 'children'),
    Input('radio1', 'value')
)



#%% Tab1
def generate_table1(selected_variable):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in [ 'Date',selected_variable]])
        ),
        html.Tbody([
            html.Tr([
                html.Td(df_test['Date'][i]),
                html.Td(df_test[selected_variable][i]),
            ]) for i in range(len(df_test))
        ])
    ])


@app.callback(
    Output('graph-container1', 'children'),
    Input('radio1', 'value')
)


def update_graph1(selected_variable):
    return dcc.Graph(
        id='graph-tab-1',
        figure={
            'data': [
                {'x': df_test['Date'], 'y': df_test['South Tower (kWh)'], 'type': 'line', 'name': 'South Tower (kWh)'},
                {'x': df_test['Date'], 'y': df_test[selected_variable], 'type': 'line', 'name': selected_variable},
            ],
            'layout': {
                'title': f'Line Chart of {selected_variable}'
            }
        }
    )

@app.callback(
    Output('update_table2', 'children'),
    Input('radio1', 'value')
)
#%% Tab 2
def update_table2(value):
    
    return (value)
@app.callback(
    Output('table-container2', 'children'),
    Input('radio2', 'value')
)

def generate_table2(selected_variable):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in [ 'Index','feature',selected_variable]])
        ),
        html.Tbody([
            html.Tr([
                html.Td(i+1), html.Td(df_select['feature'][i]),
                html.Td(df_select[selected_variable][i])
            ]) for i in range(len(df_select))
        ])
    ])

@app.callback(
    Output('graph-container2', 'children'),
    Input('radio2', 'value')
)


def update_graph2(selected_variable):
    print(222-2)
    return dcc.Graph(
        id='graph-tab-2',
        figure={
            'data': [

                {'x': df_select['feature'], 'y': df_select[selected_variable], 'type': 'bar', 'name': selected_variable},
            ],
            'layout': {
                'title': f'Line Chart of {selected_variable}'
            }
        }
    )


@app.callback(
    Output('update_table3', 'children'),
    Input('radio2', 'value')
)

#%% Tab3
def update_table3(value):
    
    return (value)
@app.callback(
    Output('table-container3', 'children'),
    Input('radio3', 'value')
)
def generate_table3(selected_variable):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in ['Index', selected_variable]])
        ),
        html.Tbody([
            html.Tr([
                html.Td(i+1), html.Td(df_metrics['Error'][i]),
                html.Td(i+1), html.Td(df_metrics[selected_variable][i])
            ]) for i in range(len(df_metrics))
        ])
    ])

@app.callback(
    Output('graph-container3', 'children'),
    Input('radio3', 'value')
)


def update_graph3(selected_variable):
    return dcc.Graph(
        id='graph-tab-3',
        figure={
            'data': [
                {'x': df_real['Date'], 'y': df_real['South Tower (kWh)'], 'type': 'line', 'name': 'real data'},
                {'x': df_forecast['Date'], 'y': df_forecast[selected_variable], 'type': 'line', 'name': selected_variable}
            ],
        
            'layout': {
                'title': f'Line Chart of {selected_variable}'
            }
        }
    )



if __name__ == '__main__':
    app.run_server(debug=True)
