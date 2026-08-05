import streamlit as st
# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Real Estate Valuation",
    layout="wide",
    page_icon="🏡"
)

if "history" not in st.session_state:
    st.session_state.history = []

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import os

openai_api_key = os.getenv("OPENAI_API_KEY")

# -----------------------------
# Load model and feature names
# -----------------------------
MODEL_PATH = os.path.join("best_model.pkl")
FEATURES_PATH = os.path.join("features.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.error("❌ Model or features file missing! Run main.py first to train and save the model.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))
feature_names = pickle.load(open(FEATURES_PATH, "rb"))  # saved list of feature names


st.markdown("""
<style>
/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #00ccff !important;
}

/* Sidebar titles */
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #00ccff !important;
}

/* Labels (sliders, inputs, etc.) */
label, .stTextInput label, .stNumberInput label {
    color: #00ccff !important;
    font-weight: bold;
}

/* Buttons */
div.stButton > button {
    background-color: black;
    color: #00ccff;
    border: 2px solid #00ccff;
    border-radius: 8px;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #00ccff;
    color: black;
}
/* Sidebar background */
section[data-testid="stSidebar"] { 
    background-color: #001f33;
    color: white;
}
</style>
""", unsafe_allow_html=True)



# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📂 Prediction History", "🤖 Smart Investment Bot", "📊 Market Insights",
     "📈 Price Trends", "📑 Comparative Market Analysis (CMA)", "ℹ️ About"]
)

# -----------------------------
# Page 1: Home (Prediction)
# -----------------------------
if page == "🏠 Home":
    st.markdown(
    """
    <h1 style='text-align: center;'>
        <span style="text-shadow: 0 0 10px #8A2BE2, 0 0 20px #8A2BE2, 0 0 40px #8A2BE2;">🏡</span> 
        <span style="background: linear-gradient(90deg, #00ccff, #66ffff); -webkit-background-clip: text; color: transparent;">
            AI-Based Real Estate Valuation System
        </span>
    </h1>
    """,
    unsafe_allow_html=True
)

    st.markdown("### Enter property details to predict the house price:")

    col1, col2 = st.columns(2)

    with col1:
        GrLivArea = st.number_input("Living Area (sqft)", min_value=300, max_value=10000, value=1500)
        OverallQual = st.slider("Overall Quality (1 = Very Poor, 10 = Excellent)", 1, 10, 5)
        OverallCond = st.slider("Overall Condition (1 = Very Poor, 10 = Excellent)", 1, 10, 5)
        YearBuilt = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)

    with col2:
        GarageCars = st.slider("Garage Capacity (Cars)", 0, 5, 2)
        FullBath = st.slider("Number of Full Bathrooms", 0, 5, 2)
        BedroomAbvGr = st.slider("Number of Bedrooms", 0, 10, 3)
        TotRmsAbvGrd = st.slider("Total Rooms Above Ground", 1, 15, 6)
        LotArea = st.number_input("Lot Area (sqft)", min_value=500, max_value=200000, value=8000)

    # Prepare raw input
    input_data = pd.DataFrame({
        'GrLivArea': [GrLivArea],
        'OverallQual': [OverallQual],
        'OverallCond': [OverallCond],
        'YearBuilt': [YearBuilt],
        'GarageCars': [GarageCars],
        'FullBath': [FullBath],
        'BedroomAbvGr': [BedroomAbvGr],
        'TotRmsAbvGrd': [TotRmsAbvGrd],
        'LotArea': [LotArea]
    })

    # Align features with training data
    input_aligned = pd.DataFrame(columns=feature_names)
    for col in input_data.columns:
        if col in input_aligned.columns:
            input_aligned[col] = input_data[col]
    input_aligned = input_aligned.fillna(0)

    if st.button("🔮 Predict House Price"):
        prediction = model.predict(input_aligned)[0]
        st.session_state.history.append(
            {**input_data.iloc[0].to_dict(), "PredictedPrice": prediction}
        )
        st.markdown(f"""
        <div style='background-color:#e6ffe6;padding:20px;border-radius:10px;border:2px solid #4CAF50'>
            <h2 style='color:#2e7d32;text-align:center;'>💰 Estimated House Price: ${prediction:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

        # Download as CSV
        csv_buffer = io.StringIO()
        pd.DataFrame([input_data.iloc[0].to_dict() | {"PredictedPrice": prediction}]).to_csv(csv_buffer, index=False)
        st.download_button("📥 Download Prediction (CSV)", data=csv_buffer.getvalue(),
                           file_name="prediction.csv", mime="text/csv")

# -----------------------------
# Page 2: Prediction History
# -----------------------------
elif page == "📂 Prediction History":
    st.title("📂 Prediction History")
    if len(st.session_state.history) == 0:
        st.info("No predictions yet. Go to Home and make one!")
    else:
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True)

        csv_buffer = io.StringIO()
        df_history.to_csv(csv_buffer, index=False)
        st.download_button("📥 Download Full History", data=csv_buffer.getvalue(),
                           file_name="prediction_history.csv", mime="text/csv")

# -----------------------------
# Page 3: Smart Investment Bot
# -----------------------------
elif page == "🤖 Smart Investment Bot":
    st.title("🤖 Smart Investment Bot")
    st.write("💡 Get AI-powered insights for smarter real estate investment decisions.")

    # Load dataset
    try:
        df = pd.read_csv("housing.csv")
    except:
        st.warning("⚠️ housing.csv not found. Please place dataset in the project folder.")
        df = None

    if df is not None:
        st.subheader("📍 Quick Insights:")
        # Existing Insights
        best_quality = df.groupby("OverallQual")["SalePrice"].mean().idxmax()
        best_year = df.groupby("YearBuilt")["SalePrice"].mean().idxmax()
        best_lot = df.loc[df["LotArea"].idxmax(), "LotArea"]
        avg_price = df["SalePrice"].mean()
        st.success(f"🏆 Best Quality Homes: **Quality {best_quality}**")
        st.success(f"📆 Best Year to Buy: **{best_year}**")
        st.success(f"🌳 Largest Lot Size Found: **{best_lot:,.0f} sqft**")
        st.success(f"💰 Average Market Price: ${avg_price:,.0f}")

        # New Features
        if "Neighborhood" in df.columns:
            best_area = df.groupby("Neighborhood")["SalePrice"].mean().idxmax()
            st.success(f"📍 Best Performing Neighborhood: **{best_area}**")

        if "YrSold" in df.columns:
            yearly_avg = df.groupby("YrSold")["SalePrice"].mean()
            growth_rate = (yearly_avg.iloc[-1] / yearly_avg.iloc[0]) ** (1/len(yearly_avg)) - 1
            st.success(f"📈 Avg Annual Growth Rate: **{growth_rate:.2%}**")

        st.subheader("💬 Ask the Investment Bot anything:")
        user_question = st.text_input("Type your question here...")

        if user_question:
            q = user_question.lower()
            # Rule-based quick answers
            if "year" in q:
                st.info(f"📆 Best year based on resale value: **{best_year}**")
            elif "quality" in q:
                st.info(f"🏆 Homes with **Overall Quality {best_quality}** perform best.")
            elif "average" in q or "price" in q:
                st.info(f"💰 The average house price is **${avg_price:,.0f}**")
            elif "lot" in q or "land" in q:
                st.info(f"🌳 The largest lot size is **{best_lot:,.0f} sqft**")
            elif "neighborhood" in q or "area" in q:
                if "Neighborhood" in df.columns:
                    st.info(f"📍 Best performing neighborhood: **{best_area}**")
            elif "growth" in q or "roi" in q:
                if "YrSold" in df.columns:
                    st.info(f"📈 Average annual growth rate: **{growth_rate:.2%}**")
            else:
                # Fallback to GPT-powered answer
                try:
                    from openai import OpenAI
                    import os
                    openai_api_key = os.environ.get("OPENAI_API_KEY")
                    if not openai_api_key:
                        st.error("❌ OpenAI API key not set in environment variables.")
                    else:
                        client = OpenAI(api_key=openai_api_key)
                        response = client.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[
                                {"role": "system", "content": "You are a helpful real estate investment assistant."},
                                {"role": "user", "content": user_question}
                            ],
                            temperature=0.7,
                            max_tokens=300
                        )
                        answer = response.choices[0].message.content
                        st.info(f"🤖 {answer}")
                except Exception as e:
                    st.error(f"❌ Error using GPT: {e}")


# -----------------------------
# Page 4: Market Insights
# -----------------------------
elif page == "📊 Market Insights":
    st.title("📊 Market Insights")
    st.write("Explore housing market trends with interactive visualizations.")

    try:
        df = pd.read_csv("housing.csv")
    except:
        st.warning("⚠️ housing.csv not found.")
        df = None

    if df is not None:
        tab1, tab2, tab3 = st.tabs(["📈 Price Distribution", "📊 Quality & Condition", "🔗 Correlation Heatmap"])
        with tab1:
            st.subheader("Distribution of House Prices")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df["SalePrice"], kde=True, bins=40, color="skyblue", ax=ax)
            st.pyplot(fig)
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                avg_price_by_quality = df.groupby("OverallQual")["SalePrice"].mean()
                fig, ax = plt.subplots()
                avg_price_by_quality.plot(kind="bar", color="orange", ax=ax)
                st.subheader("Avg Price by Quality")
                st.pyplot(fig)
            with col2:
                avg_price_by_cond = df.groupby("OverallCond")["SalePrice"].mean()
                fig, ax = plt.subplots()
                avg_price_by_cond.plot(kind="bar", color="green", ax=ax)
                st.subheader("Avg Price by Condition")
                st.pyplot(fig)
        with tab3:
            corr = df.corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
            st.pyplot(fig)

# -----------------------------
# Page 5: Price Trend Visualization
# -----------------------------
elif page == "📈 Price Trends":
    st.title("📈 Price Trend Visualization (Historical + Forecast)")

    try:
        df = pd.read_csv("housing.csv")
    except:
        st.warning("⚠️ housing.csv not found.")
        df = None

    if df is not None:
        if "YrSold" not in df.columns:
            st.error("Dataset must have 'YrSold' column for time-series trend analysis.")
        else:
            yearly_avg = df.groupby("YrSold")["SalePrice"].mean()
            st.line_chart(yearly_avg, use_container_width=True)

            # Simple Moving Average
            sma = yearly_avg.rolling(window=2).mean()

            # Linear Regression Forecast
            X = np.array(yearly_avg.index).reshape(-1, 1)
            y = yearly_avg.values
            lr = LinearRegression()
            lr.fit(X, y)
            future_years = np.arange(yearly_avg.index.min(), yearly_avg.index.max() + 3).reshape(-1, 1)
            future_preds = lr.predict(future_years)

            # ARIMA Forecast
            try:
                arima = ARIMA(y, order=(1, 1, 1))
                arima_fit = arima.fit()
                arima_forecast = arima_fit.forecast(steps=3)
            except:
                arima_forecast = [np.nan, np.nan, np.nan]

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(yearly_avg.index, yearly_avg.values, label="Historical")
            ax.plot(yearly_avg.index, sma, label="SMA (2-year)", linestyle="--")
            ax.plot(future_years.flatten(), future_preds, label="Linear Regression Forecast", linestyle=":")
            ax.legend()
            st.pyplot(fig)

            st.success(f"📊 ARIMA Forecast for next 3 years: {arima_forecast}")

# -----------------------------
# Page 6: Comparative Market Analysis (CMA)
# -----------------------------
elif page == "📑 Comparative Market Analysis (CMA)":
    st.title("📑 Comparative Market Analysis (CMA)")
    st.write("Compare multiple properties side by side and get AI-style investment suggestions.")

    uploaded = st.file_uploader("Upload CSV with property details (columns must include GrLivArea, OverallQual, SalePrice, etc.)",
                                type=["csv"])
    if uploaded:
        df_cmp = pd.read_csv(uploaded)
        st.dataframe(df_cmp, use_container_width=True)

        if st.button("🔍 Run CMA"):
            avg_price = df_cmp["SalePrice"].mean()
            best_quality = df_cmp.loc[df_cmp["SalePrice"].idxmax(), "OverallQual"]
            best_prop = df_cmp.loc[df_cmp["SalePrice"].idxmax()]

            st.success(f"🏆 Best property has Quality {best_quality} with price ${best_prop['SalePrice']:,.0f}")

            narrative = f"""
            📑 **Investment Narrative**  
            The average price of uploaded properties is ${avg_price:,.0f}.  
            The property with the highest value has **Overall Quality {best_quality}**, built in {best_prop.get('YearBuilt', 'N/A')},  
            and offers {best_prop.get('GrLivArea', 'N/A')} sqft living area.  

            ✅ Suggestion: Prioritize properties with higher quality scores and larger living area,  
            as they show stronger appreciation trends in future resale markets.
            """
            st.markdown(narrative)

# -----------------------------
# Page 7: About
# -----------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### 🏡 AI-Based Real Estate Valuation System
    This project uses **Machine Learning** to predict housing prices and provide investment insights.

    **Modules Implemented:**
    - Data Collection & Preprocessing  
    - Model Training & Evaluation  
    - Streamlit Web App (Prediction + Investment Bot + Insights + History + CMA + Trends)  

    **Extra Features Added:**
    - Prediction History with Download  
    - Investment Bot (Interactive Q&A)  
    - Correlation Heatmap & Market Trends  
    - Price Trend Forecasting (SMA, Regression, ARIMA)  
    - Comparative Market Analysis (CMA) with Narratives  

    **Developer:** 👨‍💻 Akash Pathrapalli  
    🔗 [LinkedIn](https://www.linkedin.com/in/akash-pathrapalli/)  

    © 2025 AI Real Estate Valuation System | All Rights Reserved
    """)


