import plotly.express as px

def plot_sales_trend(df):
    fig = px.line(
        df,
        x='Year_of_Release',
        y='Global_Sales',
        title='Global Sales Over Time',
        markers=True
    )
    fig.update_traces(line=dict(width=3))
    fig.update_layout(template='plotly_white')
    return fig


def plot_forecast(forecast):
    fig = px.line(
        forecast,
        x='ds',
        y='yhat',
        title='Predicted Global Sales (Prophet Forecast)',
        markers=True
    )
    fig.add_scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        name='Lower Bound'
    )
    fig.add_scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        name='Upper Bound'
    )
    fig.update_layout(template='plotly_white')
    return fig
