import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

from sklearn import linear_model





st.set_page_config(layout='wide', page_title='Startup Analysis')
df = pd.read_excel('startup_cleaned.xlsx')
# df=pd.read_csv('startup_cleaned.csv',encoding='latin-1')
df2=pd.read_excel('start.xlsx')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

colors = ["#E78CAE", "#926580", "#926580", "#707EA0", "#34495E"]
custom_palette = sns.color_palette(colors)

def predict():
    st.header("Prediction Model")
    def read_data(csv_file):
       csv_df2 = pd.read_csv(csv_file)
       return csv_df2


    def split_data(csv_df2):
       input = csv_df2.iloc[:, :-1]
       output = csv_df2.iloc[:, -1]
       x = input.values
       y = output.values.reshape((-1, 1))
       return x, y


    def find_optimize(input, outcome):
       w = np.dot(np.linalg.pinv(np.dot(input.T, input)), np.dot(input.T, outcome))
       return w


    def optimize_with_sklearn(input, outcome):
        regr = linear_model.LinearRegression(fit_intercept=False) 
        regr.fit(input, outcome)
        return regr.coef_


    def get_loss_value(input, outcome, w):
        cost = 0
        y_hat = np.dot(input, w)
        for x, y in zip(outcome, y_hat):
           print('Outcome:', x[0], 'Predict:', y[0])
           cost += pow(x[0] - y[0], 2)
           return cost / 2


    def predict_new_data(input, w):
       one = np.ones((input.shape[0], 1))
       input = np.concatenate((one, input), axis=1)
       return np.dot(input, w)


    if __name__ == '__main__':
       df2 = pd.read_excel('start.xlsx')
       st.write(df2)
       a=["Ola","BYJU'S","Zomato","CRED","PayTM",
        "Physics Wallah","PhonePe"]
       import random
       input, outcome = split_data(df2)
       one = np.ones((input.shape[0], 1))
       input = np.concatenate((one, input), axis=1)
       company=a[random.choice(range(0,7))]
       st.write("You can invest in:")
       st.write(company)


def visualize():
    




    col1,  = st.columns(1)
    with col1:
        st.header('Top 10 Startups by CAGR')
        df1= pd.read_excel('start.xlsx')
        df1 = df1.sort_values(by='CAGR', ascending=False)
        df1 = df1.head(10)
        fig, ax = plt.subplots(figsize=(25,6))

        ax.bar(df1['startup'], df1['CAGR'])

        # Set the title and labels
        ax.set_title('Top 10 Startups by CAGR')
        ax.set_xlabel('Startup Name')
        ax.set_ylabel('CAGR')
        st.pyplot(fig)
    
        st.header('Location-wise Startups')
        cityfun_series = df.groupby(['city'])['round'].count().sort_values(ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(25, 6))
        ax.bar(cityfun_series.index, cityfun_series.values)
        plt.title("Location-wise Startups", fontsize=14)
        plt.xlabel("City", fontsize=14)
        plt.ylabel("Number of startups", fontsize=14)
        st.pyplot(fig) 
        df1 = pd.read_excel('start.xlsx')

# Filter the fintech companies' valuation data to include only the fintech companies that you want to plot
        fintech_companies = ['Fin-Tech']
        df1 = df1[df1['vertical'].isin(fintech_companies)]

# Sort the filtered fintech companies' valuation data by valuation in descending order
        df1 = df1.sort_values(by=['Valuation'], ascending=False)

# Extract the company names and valuations
        company_names = df1['startup'].tolist()
        valuations = df1['Valuation'].tolist()

# Create a Matplotlib figure and axes object
        st.header("Valuation of Selected Fintech Companies")
        fig, ax = plt.subplots(figsize=(25,6))

# Plot the line chart
        ax.plot(company_names, valuations, marker='o', color='black')

# Set the title and labels
        ax.set_title('Valuation of Selected Fintech Companies')
        ax.set_xlabel('Company Name')
        ax.set_ylabel('Valuation(In USD)')

# Display the line chart in Streamlit
        st.pyplot(fig)

        st.header("CAGR vs Valuation(M$)")
        df1 = pd.read_excel('start.xlsx')
        df1 = df1.sort_values(by='CAGR', ascending=False)

        # Extract the CAGR and valuation data
        cagr = df1['CAGR'].tolist()
        valuation = df1['Valuation'].tolist()

        # Create a bar graph using the Matplotlib bar() function
        fig, ax = plt.subplots(figsize=(25,6))
        ax.bar(cagr, valuation)

        # Set the title and labels for the bar graph
        ax.set_title('CAGR vs Valuation')
        ax.set_xlabel('CAGR')
        ax.set_ylabel('Valuation')

        # Show the bar graph in Streamlit
        st.pyplot(fig)
        
        # st.header('Funding Heatmap')
        # ax = df.groupby(['year'])['vertical'].sum()
        # fig, ax = plt.subplots(figsize=(20,5))
        # # sns.heatmap()
        # # sns.heatmap(table,  cmap=custom_palette)
        # ax.set_xlabel("Year", fontsize=14)
        # ax.set_ylabel("Funding Amount", fontsize=14)
        # st.pyplot(fig)

        st.header("Sum of Valuation vs Subvertical")
        image = Image.open('img1.png')
        st.image(image,width=1000)
        st.header("Sum of CAGR vs Year")
        image2 = Image.open('img2.png')
        st.image(image2,width=1000)
        st.header("Sum of Valuation vs Year")
        image3 = Image.open('img3.png')
        st.image(image3,width=1000)
        st.header("Sum of Valuation vs Vertical")
        image4 = Image.open('img4.png')
        st.image(image4,width=1000)
        st.header("Funding for an EV Company over time evolution")
        image5 = Image.open('img5.png')
        st.image(image5,width=1000)
    
def load_overall_analysis():
    st.title('Overall Analysis')

    # total invested amount
    total = round(df['amount'].sum())
    # max amount infused in a startup
    max_funding = df.groupby('startup')['amount'].max().sort_values(ascending=False).head(1).values[0]
    # avg ticket size
    avg_funding = df.groupby('startup')['amount'].sum().mean()
    # total funded startup
    num_startups = df['startup'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Total', str(total) + 'cr')
    with col2:
        st.metric('Max', str(max_funding) + 'cr')
    with col3:
        st.metric('Avg', str(round(avg_funding)) + ' cr')
    with col4:
        st.metric('Funded Startups', num_startups)

    col1, col2 = st.columns(2)
    with col1:
        st.header('MoM graph')
        selected_option = st.selectbox('Select Type', ['Total', 'Count'])
        if selected_option == 'Total':
            temp_df = df.groupby(['year', 'month'])['amount'].sum().reset_index()
        else:
            temp_df = df.groupby(['year', 'month'])['amount'].count().reset_index()

        temp_df['x_axis'] = temp_df['month'].astype('str') + '_' + temp_df['year'].astype('str')

        # Create plot
        fig5, ax = plt.subplots()
        ax.plot(temp_df['x_axis'], temp_df['amount'])

        # Set plot labels and title
        ax.set_xlabel('Month-Month')
        ax.set_ylabel('Total Amount' if selected_option == 'Total' else 'Transaction Count')
        ax.set_title('Month-on-Month Analysis')

        # Display plot in Streamlit
        st.pyplot(fig5)

    with col2:
        st.header('Top sectors')
        sector_option = st.selectbox('Select Type ', ['Total', 'Count'])
        if sector_option == 'Total':
            tmp_df = df.groupby(['vertical'])['amount'].sum().sort_values(ascending=False).head(5)
        else:
            tmp_df = df.groupby(['vertical'])['amount'].count().sort_values(ascending=False).head(5)

        fig7, ax7 = plt.subplots()
        ax7.pie(tmp_df, labels=tmp_df.index, autopct="%0.01f%%")
        st.pyplot(fig7)

    col1, col2, = st.columns(2)
    with col1:
        st.header('Startup vs Invested Amount')
        
        cityfun_series = df.groupby(['startup'])['amount'].count().sort_values(ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.bar(cityfun_series.index, cityfun_series.values)
        plt.title("Startup vs Invested Amount", fontsize=14)
        plt.xlabel("Startup", fontsize=14)
        plt.ylabel("Investment", fontsize=14)
        st.pyplot(fig)

    with col2:
        st.header('City wise funding')
        cityfun_series = df.groupby(['city'])['round'].count().sort_values(ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.bar(cityfun_series.index, cityfun_series.values)
        plt.title("City wise funding", fontsize=14)
        plt.xlabel("City", fontsize=14)
        plt.ylabel("Number of funding", fontsize=14)
        st.pyplot(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.header('Top startups')
        top_startup = df.groupby(['startup'])['year'].count().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots()
        ax.pie(top_startup, labels=top_startup.index, autopct='%0.001f%%', shadow=True, startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    with col2:
        st.header('Top startup overall')
        overall_series = df.groupby(['startup'])['startup'].count().sort_values(ascending=False).head(8)
        fig, ax = plt.subplots()
        ax.bar(overall_series.index, overall_series.values)
        st.pyplot(fig)

    st.header('Funding Heatmap')
    table = pd.crosstab(df['year'], df['round'], values=df['amount'], aggfunc='sum')
    fig, ax = plt.subplots(figsize=(20,5))
    sns.heatmap(table,  cmap=custom_palette, vmin=0, vmax=1, annot_kws={"size": 14},
                ax=ax)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Funding Amount", fontsize=14)
    st.pyplot(fig)


def load_investor_details(investor):
    st.title(investor)
    # load the recent five investment of the investor
    lasts_5df = df[df['investors'].str.contains(investor)].head()[
        ['date', 'startup', 'vertical', 'city', 'round', 'amount']]
    st.subheader('Most Recent Investment')
    st.dataframe(lasts_5df)

    # biggest investment
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        big_series = df[df['investors'].str.contains(investor)].groupby('startup')['amount'].sum().sort_values(
            ascending=False).head()
        st.subheader('Biggest Investment')
        fig, ax = plt.subplots()
        ax.bar(big_series.index, big_series.values)
        st.pyplot(fig)

    with col2:
        vertical_series = df[df['investors'].str.contains(investor)].groupby('vertical')['amount'].sum()
        st.subheader('Sectors Invested in')
        fig1, ax1 = plt.subplots()
        ax1.pie(vertical_series, labels=vertical_series.index, autopct="%0.01f%%")
        st.pyplot(fig1)

    with col3:
        round_series = df[df['investors'].str.contains(investor)].groupby('round')['amount'].sum()
        st.subheader('Round Invested in')
        fig2, ax2 = plt.subplots()
        ax2.pie(round_series, labels=round_series.index, autopct="%0.01f%%")
        st.pyplot(fig2)

    with col4:
        city_series = df[df['investors'].str.contains(investor)].groupby('city')['amount'].sum()
        st.subheader('City Invested in')
        fig3, ax3 = plt.subplots()
        ax3.pie(city_series, labels=city_series.index, autopct="%0.01f%%")
        st.pyplot(fig3)

    col5, col6 = st.columns(2)
    with col5:
        df['Year'] = df['date'].dt.year
        year_series = df[df['investors'].str.contains(investor)].groupby('Year')['amount'].sum()
        st.subheader('YOY Investment')
        fig4, ax4 = plt.subplots()
        ax4.plot(year_series.index, year_series.values)
        st.pyplot(fig4)

    with col6:
        similar_investors = df[df['investors'].str.contains(investor)].groupby('subvertical')['amount'].sum()
        st.subheader('Similar Investor')
        fig6, ax6 = plt.subplots()
        ax6.pie(similar_investors, labels=similar_investors.index, autopct="%0.01f%%")
        st.pyplot(fig6)


def load_startup_details(startup):
    st.title(startup)
    col1, col2 = st.columns(2)
    with col1:
        # investment details
        industry_series = df[df['startup'].str.contains(startup)][['year', 'vertical', 'city', 'round']]
        st.subheader('About Startup')
        st.dataframe(industry_series)

    with col2:
        # inv_series = df[df['startup'].str.contains(startup)].groupby('investors').sum()
        inv_series = df[df['startup'].str.contains(startup)].groupby('investors')
        st.subheader('Investors')
        st.dataframe(inv_series)

    # Subindustry
    col1,col3 = st.columns(2)

    with col1:
        sub_series = df[df['startup'].str.contains(startup)].groupby('subvertical')['year'].sum()
        st.subheader('SubIndustry')
        fig9, ax9 = plt.subplots()
        ax9.pie(sub_series, labels=sub_series.index, autopct="%0.01f%%")
        st.pyplot(fig9)

    with col3:
        ver_series = df[df['startup'].str.contains(startup)].groupby('vertical')['year'].sum()
        st.subheader('Industry')
        fig10, ax10 = plt.subplots()
        ax10.pie(ver_series, labels=ver_series.index, autopct="%0.01f%%",startangle=90)
        st.pyplot(fig10)

st.sidebar.title('Startup Funding Analysis')

option = st.sidebar.selectbox('Select One',[ 'Visualization','Overall Analysis', 'Startup', 'Investor','Prediction'])

if option=='Prediction':
    predict()
elif option=='Visualization':
    visualize()

elif option == 'Overall Analysis':
    load_overall_analysis()


elif option == 'Startup':
    select_startup = st.sidebar.selectbox('Select Startup', sorted(df['startup'].unique().tolist()))
    btn1 = st.sidebar.button('Find Startup Details')
    if btn1:
        load_startup_details(select_startup)

else:
    selected_investor = st.sidebar.selectbox('Select StartUp', sorted(set(df['investors'].str.split(',').sum())))
    btn2 = st.sidebar.button('Find Investor Details')
    if btn2:
        load_investor_details(selected_investor)
