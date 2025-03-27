import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st



data = pd.read_csv("https://raw.githubusercontent.com/ikaddogh/my-data/refs/heads/main/CAR_P.csv")


df = pd.DataFrame(data)


# Group by 'make' and sum the sales (sellingprice) for each make
top_makes_by_sales = df.groupby('make')['sellingprice'].sum().sort_values(ascending=False).head(10)

# Create a bar plot for top makes by total sales
fig = px.bar(top_makes_by_sales, 
             labels={'index': 'Make', 'sellingprice': 'Total Sales'},
             title='Top Cars Made by Sales', 
             width=600, 
             height=400)

# Update axis titles
fig.update_layout(xaxis_title='Car Make', yaxis_title='Total Sales')

# Streamlit Layout
st.title('Car Sales Analysis')  # Title of the Streamlit app
st.subheader('Top Car Makes by Total Sales')  # Subheader description
st.plotly_chart(fig)  # Render the Plotly chart in Streamlit

import streamlit as st

st.write("This chart allows us to see how the mileage of a car correlates with its price. Trend of Price with Mileage typically, as a car's mileage increases, its selling price tends to decrease due to wear and tear, which reduces the car's value. The chart should reflect this, where you may see higher prices for cars in the lower odometer ranges. This analysis shows how the mileage of a car (represented by the odometer) impacts its selling price. It is expected that, generally, cars with lower mileage will have higher average selling prices, while cars with higher mileage will show a decrease in their average prices. The bar chart provides a clear, visual representation of this relationship, making it easier to identify patterns and insights from the data!")



# Assuming 'df' is your cleaned DataFrame (replace this with your actual DataFrame loading process)
# df = pd.read_csv("path_to_your_data.csv")  # Example to load your data

# Group by 'make', 'model', and 'body' and sum the sales (sellingprice) for each combination
sales_breakdown = df.groupby(['make', 'model', 'body'])['sellingprice'].sum().reset_index()

# Create a 3D scatter plot or a bubble chart for sales breakdown
fig = px.scatter(sales_breakdown, 
                 x='make', y='model', 
                 size='sellingprice', color='body', 
                 hover_name='body',
                 size_max=60, 
                 title='Sales Breakdown by Make, Model, and Body Type',
                 labels={'sellingprice': 'Total Sales'},
                 width=800, height=600)

# Update axis titles and layout
fig.update_layout(xaxis_title='Car Make', yaxis_title='Car Model', 
                  legend_title='Body Type', showlegend=True)

# Streamlit Layout
st.title('Car Sales Breakdown')  # Title of the Streamlit app
st.subheader('Sales Breakdown by Make, Model, and Body Type')  # Subheader description
st.plotly_chart(fig)  # Render the Plotly chart in Streamlit

import streamlit as st
st.write("The chart shows that better condition cars from the top 10 makes tend to have higher resale values. Condition of the car plays a major role in determining its selling price. Brand reputation can influence how condition impacts the selling price, with some makes showing larger differences in price based on condition than others. This analysis can be useful for both buyers who are looking for cars in specific conditions and sellers who want to price their cars appropriately based on their condition and make!")


import streamlit as st
import pandas as pd
import plotly.express as px

# Load your cleaned DataFrame (replace this with the actual loading process)
# df = pd.read_csv("path_to_your_data.csv")

# Assuming 'df' is already cleaned and contains 'odometer' and 'sellingprice' columns

# Define odometer bins
odometer_bins = [0, 50000, 100000, 150000, 200000, 250000, float('inf')]  # Adjust the ranges as needed
odometer_labels = ['0-50K', '50K-100K', '100K-150K', '150K-200K', '200K-250K', '250K+']

# Create a new column for the odometer range
df['odometer_range'] = pd.cut(df['odometer'], bins=odometer_bins, labels=odometer_labels, right=False)

# Group by the odometer range and calculate the average selling price for each range
selling_price_by_odometer = df.groupby('odometer_range')['sellingprice'].mean().reset_index()

# Create a bar plot using Plotly
fig = px.bar(selling_price_by_odometer, 
             x='odometer_range', y='sellingprice', 
             title='Average Selling Price by Odometer Range',
             labels={'odometer_range': 'Odometer Range', 'sellingprice': 'Average Selling Price'},
             color='odometer_range', 
             color_continuous_scale='Viridis',
             width=800, height=600)

# Update the layout with axis titles
fig.update_layout(xaxis_title='Odometer Range', yaxis_title='Average Selling Price')

# Streamlit Layout
st.title('Car Selling Price Analysis')  # Title of the Streamlit app
st.subheader('Average Selling Price by Odometer Range')  # Subheader description
st.plotly_chart(fig)  # Render the Plotly chart in Streamlit

import streamlit as st
st.write("A positive correlation means that as one feature increases, the other feature also increases. A negative value correlation indicates that as one feature increases, the other decreases. A value zero ( 0) indicates no linear relationship between the two features. The correlation matrix helps us understand which numerical features are strongly correlated with each other and which are independent. This information is crucial for feature selection in predictive models or simply for understanding how different aspects of the car relate to its price or other numerical factors in the dataset!")


import streamlit as st
import pandas as pd
import plotly.express as px

# Load your cleaned DataFrame (replace this with the actual loading process)
# df = pd.read_csv("path_to_your_data.csv")

# Assuming 'df' is already cleaned and contains 'make', 'condition', and 'sellingprice' columns

# Step 1: Find the top 10 makes by sales (or you could use another metric)
top_makes = df['make'].value_counts().head(10).index.tolist()

# Step 2: Filter the dataset to include only the top makes
df_top_makes = df[df['make'].isin(top_makes)]

# Step 3: Group by 'make' and 'condition' and calculate the average selling price
condition_price = df_top_makes.groupby(['make', 'condition'])['sellingprice'].mean().reset_index()

# Step 4: Create a bar plot showing the impact of condition on selling price for each make
fig = px.bar(condition_price, 
             x='condition', y='sellingprice', 
             color='make', 
             title='Impact of Condition on Selling Price within Top Makes',
             labels={'condition': 'Condition', 'sellingprice': 'Average Selling Price', 'make': 'Car Make'},
             barmode='group',  # To group bars for each make
             height=600, width=800)

# Step 5: Customize the layout
fig.update_layout(xaxis_title='Condition', yaxis_title='Average Selling Price',
                  legend_title='Car Make')

# Streamlit Layout
st.title('Car Selling Price Analysis')  # Main title of the app
st.subheader('Impact of Condition on Selling Price within Top Makes')  # Subheader
st.plotly_chart(fig)  # Display the plotly bar chart in Streamlit


import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load your cleaned DataFrame (replace this with the actual loading process)
# df = pd.read_csv("path_to_your_data.csv")

# Assuming 'df' is the DataFrame with 'sellingprice' and other numerical features

# Streamlit Layout
st.title('Correlation Matrix of Numerical Features')

# Step 1: Select only numerical columns for correlation calculation
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Step 2: Compute the correlation matrix
correlation_matrix = df[numerical_columns].corr()

# Step 3: Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Add title
plt.title('Correlation Matrix of Numerical Features', fontsize=16)

# Step 4: Show the plot in Streamlit
st.pyplot(plt)  # Display the plot in the Streamlit app

import streamlit as st
st.write("Overall, the condition of the car is a key factor in determining the selling price, with better condition cars fetching higher prices. However, the impact of condition might vary by car make, with some makes being more sensitive to condition than others!")


import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'df' is your cleaned DataFrame

# Streamlit Layout
st.title('Selling Price Distribution by Top Car Makes')

# Step 1: Calculate the top 10 makes by the number of cars sold
top_makes = df['make'].value_counts().head(10).index.tolist()

# Step 2: Filter the DataFrame to include only the rows for top makes
top_makes_df = df[df['make'].isin(top_makes)]

# Step 3: Create a boxplot or violin plot for selling price distribution by make
plt.figure(figsize=(12, 8))
sns.boxplot(data=top_makes_df, x='make', y='sellingprice', order=top_makes, palette='coolwarm')

# Step 4: Add title and labels
plt.title('Selling Price Distribution by Top Car Makes', fontsize=16)
plt.xlabel('Car Make', fontsize=12)
plt.ylabel('Selling Price ($)', fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Step 5: Show the plot in Streamlit
st.pyplot(plt)  # Display the plot in the Streamlit app

import streamlit as st
st.write("The chart shows that certain car colors have a higher average selling price than others. The bar heights show which colors are associated with higher or lower prices. Popular car colors, like black or white, might tend to have a higher average price, potentially because these colors are more in demand. A taller bar means that the average selling price for that color is higher, indicating a potential market preference for that specific color!")


import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'df' is your cleaned DataFrame

# Streamlit Layout
st.title('Scatter Plot Matrix of Numerical Features')

# List of numerical features (adjust based on your dataset)
numerical_columns = ['sellingprice', 'odometer', 'year', 'mmr', 'condition']  # Add more columns if necessary

# Step 1: Create a scatter plot matrix (pairplot) for the numerical features
sns.pairplot(df[numerical_columns])

# Step 2: Set the title of the plot
plt.suptitle('Scatter Plot Matrix of Numerical Features', y=1.02)

# Step 3: Show the plot in Streamlit
st.pyplot(plt)  # Display the plot in the Streamlit app

import streamlit as st
import plotly.express as px
import pandas as pd

# Example data for testing
data = {
    'make': ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Toyota', 'Ford', 'Chevrolet', 'BMW', 'Honda'],
    'sellingprice': [20000, 22000, 25000, 27000, 50000, 21000, 24000, 26000, 52000, 23000]
}

df = pd.DataFrame(data)  # Replace this with actual loading of your data

# Streamlit Layout
st.title('Car Sales Analysis')
st.subheader('Interactive Top Car Makes by Total Sales')

# Group by 'make' and sum the sales (sellingprice) for each make
top_makes_by_sales = df.groupby('make')['sellingprice'].sum().sort_values(ascending=False).head(10)

# Create a bar plot for top makes by total sales
fig = px.bar(top_makes_by_sales, 
             labels={'index': 'Make', 'sellingprice': 'Total Sales'},
             title='Top Cars Made by Sales', 
             width=600, 
             height=400)

# Update axis titles
fig.update_layout(xaxis_title='Car Make', yaxis_title='Total Sales')

# Add interactivity features:
# Enabling hover data to show the exact sales number when hovering over bars
fig.update_traces(hoverinfo='x+y+text', hoverlabel=dict(bgcolor="white", font_size=13))

# Enabling zooming and panning on the chart by default
fig.update_layout(
    dragmode='zoom',  # Zooming enabled by default
    hovermode='closest',  # Display data for closest bar during hover
    xaxis=dict(tickangle=45)  # Rotate x-axis labels for readability
)

# Show the Plotly chart in Streamlit
st.plotly_chart(fig)

# Add an option to show raw data
if st.checkbox('Show Raw Data'):
    st.write(top_makes_by_sales)

import streamlit as st
st.write("This chart allows us to see how the mileage of a car correlates with its price. Trend of Price with Mileage typically, as a car's mileage increases, its selling price tends to decrease due to wear and tear, which reduces the car's value. The chart should reflect this, where you may see higher prices for cars in the lower odometer ranges. This analysis shows how the mileage of a car (represented by the odometer) impacts its selling price. It is expected that, generally, cars with lower mileage will have higher average selling prices, while cars with higher mileage will show a decrease in their average prices. The bar chart provides a clear, visual representation of this relationship, making it easier to identify patterns and insights from the data!")

    

import streamlit as st
import plotly.express as px
import pandas as pd

# Example DataFrame (replace with your actual data)
# df = pd.read_csv("your_data.csv")  # Uncomment this if you load the data from a CSV
data = {
    'make': ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Toyota', 'Ford', 'Chevrolet', 'BMW', 'Honda'],
    'sellingprice': [20000, 22000, 25000, 27000, 50000, 21000, 24000, 26000, 52000, 23000]
}
df = pd.DataFrame(data)  # Replace this with actual loading of your data

# Streamlit Layout
st.title('Car Sales Analysis')
st.subheader('Interactive Top Car Makes by Total Sales')

# Dropdown to select top N car makes
top_n = st.slider('Select the number of top car makes to display', 1, 20, 10)

# Group by 'make' and sum the sales (sellingprice) for each make
top_makes_by_sales = df.groupby('make')['sellingprice'].sum().sort_values(ascending=False).head(top_n)

# Create a bar plot for top makes by total sales
fig = px.bar(top_makes_by_sales, 
             labels={'index': 'Make', 'sellingprice': 'Total Sales'},
             title=f'Top {top_n} Cars Made by Sales', 
             width=600, 
             height=400)

# Update axis titles
fig.update_layout(xaxis_title='Car Make', yaxis_title='Total Sales')

# Show the Plotly chart in Streamlit
st.plotly_chart(fig)

# Add a checkbox to display the data behind the plot
if st.checkbox('Show raw data'):
    st.write(top_makes_by_sales)

