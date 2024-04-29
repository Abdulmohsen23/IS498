import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import markdown
import google.generativeai as genai
from sklearn.preprocessing import LabelEncoder


# Load your DataFrame (replace the path with your actual CSV file)
df = pd.read_csv(r"C:\Users\osama\Documents\IS498-ML\data\Equites_Historical_Adjusted_Prices_Report.csv", index_col=False)

def to_markdown(text):
   # Replace the special character with a Markdown bullet point
    text = text.replace('•', ' *')
    return text

GOOGLE_API_KEY = 'AIzaSyDgIwUdoRxZ-mPPZcOwpRRemunKadXhFAU'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.0-pro')

# Function to process an industry-year and get sentiments 
def process_industry_year(df, industry, year):
    # Filter data for the specific industry and year
    data = df[(df['Industry Group'] == industry) & (df['Year'] == year)]

    # Construct the first prompt (you can customize this further)
    prompt1 = f"For the Saudi Arabian {industry} industry in {year}, what was the general state of it over each qurter of the year?"
    response1 = model.generate_content(prompt1)
    general_state = to_markdown(response1.text)
    print(general_state)
    print('--------------')  

    # Analyze the first response for sentiment
    prompt2 = f"Considering the information from this article {general_state}, was the state of the {industry} industry over each qurter in {year}, is it very positive or positive or very negative or negative or neutral? Just answer with these 5 words in this format(Q1 state, Q2 state. Q3 state ,Q4 state ) and do not write anything else or change this format"
    response2 = model.generate_content(prompt2)
    sentiment = to_markdown(response2.text)
    print(sentiment)
    print('-----------')

    return general_state, sentiment

# Function to extract year from date column
df['Year'] = pd.to_datetime(df['Date']).dt.year

# Add new columns to your DataFrame
df['Industry_State_General'] = ""
df['Industry_State_Sentiment'] = ""



# Iterate through unique industry-year combinations
for industry in df['Industry Group'].unique():
    for year in range(2010, 2025):  # Generate prompts from 2010 to 2024
        general_state, sentiment = process_industry_year(df.copy(), industry, year)  

        df.loc[(df['Industry Group'] == industry) & (df['Year'] == year), 'Industry_State_General'] = general_state
        df.loc[(df['Industry Group'] == industry) & (df['Year'] == year), 'Industry_State_Sentiment'] = sentiment

# (Optional) Save the updated DataFrame
df.to_csv("updated_data.csv", index=False)

print(df) 


