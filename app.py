import streamlit as st
import numpy as np
import joblib
import pickle
import nltk
import pandas as pd
from sklearn.datasets import fetch_california_housing
import plotly.express as px
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config ONCE
st.set_page_config(page_title="House Price Prediction App",layout = "wide")

mainSection = st.container()

# ----------------- MAIN PAGE -----------------
def show_main_page():
    with mainSection:

        # --- Enhanced Professional Styling ---
        st.markdown("""
            <style>
            .stApp {
                background-image: url("https://i.ibb.co/QjQHvRZf/image.jpg"); 
                background-size: cover; 
                background-repeat: no-repeat;
                background-attachment: fixed; 
                position: relative;
            }
            .stApp::before {
                content: "";
                position: absolute;
                top: 0; left: 0; right: 0; bottom: 0;
                background-color: rgba(0, 0, 0, 0.55);
                z-index: 0;
            }
            .stApp > div { position: relative; z-index: 1; color: white; }

            h1, h2, h3, p, label { color: white !important; }

            /* Enhanced Button Styling */
            div.stButton > button:first-child {
                background: linear-gradient(135deg, #00BFFF 0%, #1E90FF 50%, #4169E1 100%);
                color: white;
                border-radius: 15px;
                height: 3.5em;
                width: 100%;
                font-weight: bold;
                font-size: 1.1rem;
                border: 2px solid rgba(255,255,255,0.3);
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,191,255,0.3);
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            div.stButton > button:first-child:hover {
                background: linear-gradient(135deg, #1E90FF 0%, #4169E1 50%, #0000CD 100%);
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,191,255,0.5);
                border-color: #00FFFF;
            }
            
            /* Primary Button Special Styling */
            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #FF6B35 0%, #F7931E 50%, #FFD700 100%) !important;
                border: 2px solid rgba(255,255,255,0.4) !important;
                box-shadow: 0 4px 15px rgba(255,107,53,0.4) !important;
            }
            .stButton > button[kind="primary"]:hover {
                background: linear-gradient(135deg, #F7931E 0%, #FFD700 50%, #FFA500 100%) !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(255,107,53,0.6) !important;
            }
                    
            /* Download Button Styling */
            div.stDownloadButton > button:first-child {
                background: linear-gradient(135deg, #32CD32 0%, #228B22 50%, #006400 100%);
                color: white;
                border-radius: 12px;
                height: 3em;
                width: 100%;
                font-weight: bold;
                border: 2px solid rgba(255,255,255,0.3);
                transition: all 0.3s ease;
                box-shadow: 0 4px 12px rgba(50,205,50,0.3);
            }
            div.stDownloadButton > button:first-child:hover {
                background: linear-gradient(135deg, #228B22 0%, #006400 50%, #004000 100%);
                transform: translateY(-1px);
                box-shadow: 0 6px 16px rgba(50,205,50,0.5);
            }

            /* Professional Expander Styling */
            .streamlit-expanderHeader {
                background: linear-gradient(90deg, rgba(0,191,255,0.1), rgba(30,144,255,0.1)) !important;
                border: 1px solid rgba(0,191,255,0.3) !important;
                border-radius: 10px !important;
            }

            /* Enhanced Metric Styling for Better Visibility */
            [data-testid="metric-container"] {
                background: linear-gradient(135deg, rgba(0,191,255,0.1), rgba(30,144,255,0.1));
                border: 1px solid rgba(0,191,255,0.3);
                border-radius: 12px;
                padding: 15px;
                box-shadow: 0 2px 8px rgba(0,191,255,0.2);
            }
            
            [data-testid="metric-container"] > div > div:first-child {
                color: #00BFFF !important;
                font-weight: bold !important;
                font-size: 1rem !important;
            }
            
            [data-testid="metric-container"] > div > div:nth-child(2) {
                color: #FFFFFF !important;
                font-weight: bold !important;
                font-size: 1.8rem !important;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            }
            
            [data-testid="metric-container"] > div > div:nth-child(3) {
                color: #90EE90 !important;
                font-weight: bold !important;
                font-size: 1rem !important;
            }

            /* Individual Model Color Styling */
            .linear-model [data-testid="metric-container"] {
                background: linear-gradient(135deg, rgba(65,105,225,0.15), rgba(30,144,255,0.15)) !important;
                border: 2px solid rgba(65,105,225,0.4) !important;
            }
            .linear-model [data-testid="metric-container"] > div > div:nth-child(2) {
                color: #4169E1 !important;
                text-shadow: 1px 1px 3px rgba(0,0,0,0.7) !important;
            }

            .ridge-model [data-testid="metric-container"] {
                background: linear-gradient(135deg, rgba(138,43,226,0.15), rgba(75,0,130,0.15)) !important;
                border: 2px solid rgba(138,43,226,0.4) !important;
            }
            .ridge-model [data-testid="metric-container"] > div > div:nth-child(2) {
                color: #8A2BE2 !important;
                text-shadow: 1px 1px 3px rgba(0,0,0,0.7) !important;
            }

            .lasso-model [data-testid="metric-container"] {
                background: linear-gradient(135deg, rgba(255,140,0,0.15), rgba(255,69,0,0.15)) !important;
                border: 2px solid rgba(255,140,0,0.4) !important;
            }
            .lasso-model [data-testid="metric-container"] > div > div:nth-child(2) {
                color: #FF8C00 !important;
                text-shadow: 1px 1px 3px rgba(0,0,0,0.7) !important;
            }
            </style>
            """, unsafe_allow_html=True)


        # --- Professional Header ---
        st.markdown("""
            <div style='text-align: center; padding: 30px 0;'>
                <h1 style='color: #00BFFF; font-size: 3rem; margin-bottom: 10px;'>🏡 California Real Estate Analytics Platform</h1>
                <h3 style='color: white; font-weight: 300; margin-bottom: 5px;'>Advanced Machine Learning Property Valuation System</h3>
                <p style='color: #B0C4DE; font-size: 1.1rem;'>Powered by Linear, Ridge & Lasso Regression Models | Developed by Sanjana</p>
            </div>
        """, unsafe_allow_html=True)

        # Load the dataset
        try:
          df = pd.read_csv("california_housing.csv")
        except Exception as e:
          st.error("Dataset not found. Please check file path.")
          st.stop()

        # --- Load Resources ---
        with open("corpus.pkl", "rb") as f:
            qa_corpus = pickle.load(f)

        @st.cache_resource
        def download_nltk_data():
            nltk.download("punkt")
            nltk.download('punkt_tab')
            nltk.download("wordnet")
            nltk.download("omw-1.4")
            nltk.download("averaged_perceptron_tagger")
        
        download_nltk_data()

        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        # Load models
        lr = joblib.load("linear_model.pkl")
        ridge = joblib.load("ridge_model.pkl")
        lasso = joblib.load("lasso_model.pkl")
        scaler = joblib.load("scaler.pkl")

        # --- TF-IDF Setup ---
        questions = [q for q, _ in qa_corpus]
        vectorizer = TfidfVectorizer(stop_words='english', analyzer='char', ngram_range=(3, 5))
        corpus_vectors = vectorizer.fit_transform(questions)

        # --- NLP Preprocessing ---
        def process_text(user_query):
            tokens = [w for w in word_tokenize(user_query.lower()) if w.isalnum()]
            stemmed = [stemmer.stem(w) for w in tokens]
            lemmatized = [lemmatizer.lemmatize(w, pos='v') for w in tokens]
            pos_tags = pos_tag(tokens)
            bigrams = list(ngrams(tokens, 2))
            trigrams = list(ngrams(tokens, 3))
            return stemmed, lemmatized, pos_tags, bigrams, trigrams

        # --- Smart Chatbot ---
        def get_response(user_query):
            query_vector = vectorizer.transform([user_query])
            sims = cosine_similarity(query_vector, corpus_vectors).flatten()
            best_idx = np.argmax(sims)
            best_score = sims[best_idx]

            if best_score > 0.1:
                answer = qa_corpus[best_idx][1]
            else:
                answer = "Sorry, I don't have an answer for that. Please ask another question!"

            return answer, best_score

        # --- Dataset Information Section ---
        st.markdown("---")
        with st.expander("📊 **California Housing Dataset - Comprehensive Analytics**", expanded=False):
            st.markdown("""
                <div style='background: linear-gradient(90deg, rgba(0,191,255,0.1), rgba(30,144,255,0.1)); 
                           padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                    <h4 style='color: #00BFFF; text-align: center; margin-bottom: 15px;'>
                        📈 Dataset Intelligence Dashboard
                    </h4>
                </div>
            """, unsafe_allow_html=True)
            
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.markdown("### 📊 **Dataset Metrics**")
                st.metric(
                    label="📋 Total Records", 
                    value=f"{df.shape[0]:,}",
                    help="Number of housing records in the dataset"
                )
                st.metric(
                    label="🏷️ Features", 
                    value=df.shape[1],
                    help="Number of property characteristics analyzed"
                )
                st.metric(
                    label="💾 Data Points", 
                    value=f"{df.size:,}",
                    help="Total data elements processed"
                )
            
            with col_info2:
                st.markdown("### 🏠 **Sample Data Preview**")
                st.markdown("*First 5 records from the California housing dataset*")
                st.dataframe(
                    df.head(), 
                    use_container_width=True,
                    height=200
                )
            
            with col_info3:
                st.markdown("### 📈 **Statistical Analysis**")
                st.markdown("*Descriptive statistics for all numerical features*")
                st.dataframe(
                    df.describe().round(2), 
                    use_container_width=True,
                    height=200
                )

        # --- AI Assistant Section ---
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; padding: 25px 0;'>
                <h2 style='color: #00BFFF; margin-bottom: 10px; font-size: 2.2rem;'>
                    🤖 AI Property Intelligence Assistant
                </h2>
                <p style='color: #B0C4DE; font-size: 1.2rem; margin-bottom: 5px;'>
                    Advanced Natural Language Processing for Real Estate Insights
                </p>
                <p style='color: white; font-size: 1rem;'>
                    Ask questions about market trends, property analysis, and housing predictions
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Dropdown for predefined questions
        dropdown_question = st.selectbox(
            "Select a predefined question (optional):",
            ["-- Select a question --"] + questions
        )

        # Text input for custom question
        user_question = st.text_input("Or type your own question:")

        # Determine which question to use (user input takes priority)
        final_question = None
        if user_question.strip():
            final_question = user_question.strip()
        elif dropdown_question != "-- Select a question --":
            final_question = dropdown_question

        if st.button("Get Answer"):
            if final_question:
                answer, score = get_response(final_question)
                stemmed, lemmatized, pos_tags, bigrams, trigrams = process_text(final_question)

                st.success(f"Answer: {answer}")
            else:
                st.warning("Please select or type a question!")

        # --- FAQs Section ---

            # Add a fake categorical column for demo (since dataset is numeric)
        df["OceanProximity"] = np.random.choice(
            ["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN"], size=len(df)
        )

        # --- FAQs Section ---
        st.subheader("❓ Frequently Asked Questions")

        # Q1: House Age Distribution (Histogram)
        with st.expander("Q1: What is the distribution of house age?"):
            st.write("**Analysis:** The data shows houses range from 1 to 52 years old. Most properties are concentrated between 10-40 years, with peak frequency around 15-25 years. This indicates a significant portion of California housing stock was built in the 1980s-1990s.")
            fig1 = px.histogram(
                df, x="HouseAge", nbins=25, 
                title="House Age Distribution Analysis",
                labels={"HouseAge": "House Age (Years)", "count": "Number of Properties"},
                color_discrete_sequence=["#2E86AB"]
            )
            fig1.update_layout(
                title_font_size=18,
                title_x=0.5,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=14),
                xaxis=dict(
                    gridcolor='rgba(255,255,255,0.2)',
                    tickfont=dict(size=14, color='white'),
                    title_font=dict(size=16, color='white')
                ),
                yaxis=dict(
                    gridcolor='rgba(255,255,255,0.2)',
                    tickfont=dict(size=14, color='white'),
                    title_font=dict(size=16, color='white')
                )
            )
            st.plotly_chart(fig1, use_container_width=True)

        # Q2: Median Income Distribution (Histogram)
        with st.expander("Q2: What is the distribution of median income?"):
            st.write("**Analysis:** Median household income ranges from $5,000 to $150,000+ annually. The distribution is right-skewed with most households earning $20,000-$60,000. Higher income areas (>$80,000) are less common, indicating income inequality across California regions.")
            fig2 = px.histogram(
                df, x="MedInc", nbins=25, 
                title="Median Income Distribution Analysis",
                labels={"MedInc": "Median Income (10k USD)", "count": "Number of Households"},
                color_discrete_sequence=["#A23B72"]
            )
            fig2.update_layout(
                title_font_size=18,
                title_x=0.5,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=14),
                xaxis=dict(
                    gridcolor='rgba(255,255,255,0.2)',
                    tickfont=dict(size=14, color='white'),
                    title_font=dict(size=16, color='white')
                ),
                yaxis=dict(
                    gridcolor='rgba(255,255,255,0.2)',
                    tickfont=dict(size=14, color='white'),
                    title_font=dict(size=16, color='white')
                )
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Q3: Ocean Proximity (Pie Chart)
        with st.expander("Q3: How are houses distributed by ocean proximity?"):
            st.write("**Analysis:** This synthetic data shows housing distribution by coastal access. In reality, 'INLAND' properties typically dominate California housing stock, while '<1H OCEAN' and 'NEAR OCEAN' properties command premium prices due to desirable coastal locations.")
            fig3 = px.pie(
                df, names="OceanProximity", 
                title="Property Distribution by Ocean Proximity",
                color_discrete_sequence=["#F18F01", "#C73E1D", "#2E86AB", "#A23B72"]
            )
            fig3.update_layout(
                title_font_size=18,
                title_x=0.5,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=14),
                showlegend=True
            )
            fig3.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                textfont_size=14
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Q4: Average Rooms per Household (Bar Chart)
        with st.expander("Q4: What is the average number of rooms per house age group?"):
            st.write("**Analysis:** Newer homes (built more recently) tend to have more rooms on average, reflecting modern housing trends toward larger living spaces. Older homes show fewer rooms, consistent with historical building patterns and smaller family sizes in earlier decades.")
            df["AgeGroup"] = pd.cut(df["HouseAge"], bins=5)
            df_bar = df.groupby("AgeGroup")["AveRooms"].mean().reset_index()
            df_bar["AgeGroup"] = df_bar["AgeGroup"].astype(str)
            fig4 = px.bar(
                df_bar, x="AgeGroup", y="AveRooms", 
                title="Average Rooms by House Age Groups",
                labels={"AgeGroup": "House Age Groups", "AveRooms": "Average Rooms per House"},
                color_discrete_sequence=["#F18F01"]
            )
            fig4.update_layout(
                title_font_size=18,
                title_x=0.5,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=14),
                xaxis=dict(
                    gridcolor='rgba(255,255,255,0.2)',
                    tickfont=dict(size=14, color='white'),
                    title_font=dict(size=16, color='white')
                ),
                yaxis=dict(
                    gridcolor='rgba(255,255,255,0.2)',
                    tickfont=dict(size=14, color='white'),
                    title_font=dict(size=16, color='white')
                )
            )
            fig4.update_traces(marker_line_color='white', marker_line_width=1)
            st.plotly_chart(fig4, use_container_width=True)

        # Q5: Population Distribution (Histogram)
        with st.expander("Q5: What is the population distribution?"):
            st.write("**Analysis:** Block-level population shows high variability from sparse rural areas (100-500 people) to dense urban blocks (3000+ people). The right-skewed distribution indicates most blocks have moderate population density (500-2000), with fewer high-density urban centers.")
            fig5 = px.histogram(
                df, x="Population", nbins=25, 
                title="Population Distribution Analysis",
                labels={"Population": "Population per Block", "count": "Number of Blocks"},
                color_discrete_sequence=["#C73E1D"]
            )
            fig5.update_layout(
                title_font_size=18,
                title_x=0.5,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=14),
                xaxis=dict(
                    gridcolor='rgba(255,255,255,0.2)',
                    tickfont=dict(size=14, color='white'),
                    title_font=dict(size=16, color='white')
                ),
                yaxis=dict(
                    gridcolor='rgba(255,255,255,0.2)',
                    tickfont=dict(size=14, color='white'),
                    title_font=dict(size=16, color='white')
                )
            )
            st.plotly_chart(fig5, use_container_width=True)
        # --- Enhanced Input Section ---
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; padding: 20px;'>
                <h2 style='color: #00BFFF; margin-bottom: 10px;'>🏠 Property Prediction Calculator</h2>
                <p style='color: white; font-size: 16px;'>Enter property details to get AI-powered price predictions</p>
            </div>
        """, unsafe_allow_html=True)

        # Enhanced Input fields with better styling
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("### 💰 **Economic Factors**")
            MedInc = st.slider("💵 Median Income (10k USD)", min_value=0.0, max_value=15.0, value=5.0, step=0.1, 
                              help="Median household income in the area (in tens of thousands)")
            
            st.markdown("### 🏘️ **Property Details**")
            HouseAge = st.slider("📅 House Age (Years)", min_value=1.0, max_value=52.0, value=20.0, step=1.0,
                                help="Age of the property in years")
            AveRooms = st.slider("🏠 Average Rooms", min_value=1.0, max_value=10.0, value=5.0, step=0.1,
                               help="Average number of rooms per household")
            AvgBedrms = st.slider("🛏️ Average Bedrooms", min_value=0.0, max_value=5.0, value=1.0, step=0.1,
                                help="Average number of bedrooms per household")

        with col2:
            st.markdown("### 👥 **Demographics**")
            Population = st.slider("🏘️ Block Population", min_value=1.0, max_value=5000.0, value=1000.0, step=50.0,
                                  help="Total population in the census block")
            AveOccup = st.slider("👨‍👩‍👧‍👦 Average Occupancy", min_value=1.0, max_value=10.0, value=3.0, step=0.1,
                               help="Average number of people per household")
            
            st.markdown("### 📍 **Location**")
            Latitude = st.slider("🌍 Latitude", min_value=32.0, max_value=42.0, value=36.0, step=0.1,
                                help="Geographic latitude coordinate")
            Longitude = st.slider("🌍 Longitude", min_value=-124.0, max_value=-114.0, value=-119.0, step=0.1,
                                 help="Geographic longitude coordinate")

        # Prepare features
        features = np.array([[MedInc, HouseAge, AveRooms, AvgBedrms, Population, AveOccup, Latitude, Longitude]])
        features_scaled = scaler.transform(features)

        # Enhanced Prediction Section
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; padding: 25px; background: linear-gradient(90deg, rgba(0,191,255,0.1), rgba(30,144,255,0.1)); 
                       border-radius: 15px; margin: 20px 0;'>
                <h2 style='color: #00BFFF; margin-bottom: 10px; font-size: 2rem;'>🔮 AI Property Valuation Engine</h2>
                <p style='color: white; font-size: 1.1rem;'>Advanced machine learning models ready to analyze your property</p>
            </div>
        """, unsafe_allow_html=True)

        # Professional prediction button
        col_pred1, col_pred2, col_pred3 = st.columns([1, 1, 1])
        
        with col_pred2:
            predict_button = st.button(
                "🚀 GENERATE PROPERTY VALUATION", 
                type="primary",
                use_container_width=True,
                help="Click to run AI analysis on your property parameters"
            )

        # Show predictions when button is clicked
        if predict_button:
            # Calculate predictions
            lr_pred = lr.predict(features_scaled)[0] * 100000
            ridge_pred = ridge.predict(features_scaled)[0] * 100000
            lasso_pred = lasso.predict(features_scaled)[0] * 100000
            avg_pred = (lr_pred + ridge_pred + lasso_pred) / 3
            
            # Store predictions in session state for download
            st.session_state['predictions'] = {
                'inputs': {
                    'Median Income': f"${MedInc * 10000:,.0f}",
                    'House Age': f"{HouseAge} years",
                    'Average Rooms': f"{AveRooms:.1f}",
                    'Average Bedrooms': f"{AvgBedrms:.1f}",
                    'Population': f"{Population:,.0f}",
                    'Average Occupancy': f"{AveOccup:.1f}",
                    'Latitude': f"{Latitude:.2f}",
                    'Longitude': f"{Longitude:.2f}"
                },
                'predictions': {
                    'Linear Regression': f"${lr_pred:,.0f}",
                    'Ridge Regression': f"${ridge_pred:,.0f}",
                    'Lasso Regression': f"${lasso_pred:,.0f}",
                    'Average Prediction': f"${avg_pred:,.0f}"
                }
            }

        # Display results if predictions exist
        if 'predictions' in st.session_state:
            st.markdown("---")
            st.markdown("""
                <div style='text-align: center; padding: 15px; background: linear-gradient(90deg, rgba(0,191,255,0.1), rgba(30,144,255,0.1)); 
                           border-radius: 10px; margin: 15px 0;'>
                    <h2 style='color: #00BFFF; margin: 0;'>📊 AI Valuation Results</h2>
                    <p style='color: white; margin: 5px 0;'>Professional property analysis complete</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Extract values for display
            lr_pred = float(st.session_state['predictions']['predictions']['Linear Regression'].replace('$', '').replace(',', ''))
            ridge_pred = float(st.session_state['predictions']['predictions']['Ridge Regression'].replace('$', '').replace(',', ''))
            lasso_pred = float(st.session_state['predictions']['predictions']['Lasso Regression'].replace('$', '').replace(',', ''))
            avg_pred = (lr_pred + ridge_pred + lasso_pred) / 3
            
            # Enhanced metrics display with better spacing
            col_res1, col_res2, col_res3 = st.columns(3, gap="medium")
            
            with col_res1:
                st.markdown('<div class="linear-model">', unsafe_allow_html=True)
                st.metric(
                    label="🔵 Linear Regression Model",
                    value=f"${lr_pred:,.0f}",
                    delta=f"{((lr_pred - avg_pred) / avg_pred * 100):+.1f}%",
                    help="Basic linear relationship model prediction"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_res2:
                st.metric(
                    label="🟣 Ridge Regression Model", 
                    value=f"${ridge_pred:,.0f}",
                    delta=f"{((ridge_pred - avg_pred) / avg_pred * 100):+.1f}%",
                    help="Regularized model preventing overfitting"
                )
            
            with col_res3:
                st.metric(
                    label="🟠 Lasso Regression Model",
                    value=f"${lasso_pred:,.0f}",
                    delta=f"{((lasso_pred - avg_pred) / avg_pred * 100):+.1f}%",
                    help="Feature selection with L1 regularization"
                )
            
            # Enhanced average prediction highlight
            st.markdown(f"""
                <div style='text-align: center; padding: 25px; 
                           background: linear-gradient(135deg, #FF6B35 0%, #F7931E 50%, #FFD700 100%); 
                           border-radius: 15px; margin: 25px 0; 
                           box-shadow: 0 4px 15px rgba(255,107,53,0.4);
                           border: 2px solid rgba(255,255,255,0.3);'>
                    <h1 style='color: white; margin: 0; font-size: 2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                        🎯 ${avg_pred:,.0f}
                    </h1>
                    <h3 style='color: white; margin: 10px 0; font-weight: 300;'>
                        Recommended Market Valuation
                    </h3>
                    <p style='color: rgba(255,255,255,0.9); margin: 5px 0; font-size: 1rem;'>
                        Consensus prediction from all three AI models
                    </p>
                </div>
            """, unsafe_allow_html=True)

        # Enhanced Download Section
        st.markdown("---")
        download_choice = st.radio(
            "📥 **Download Prediction Report**", 
            ("No", "Yes"),
            help="Download a detailed report of your property analysis"
        )

        if download_choice == "Yes":
            if 'predictions' in st.session_state:
                # Create meaningful download content
                report_content = f"""
CALIFORNIA HOUSE PRICE PREDICTION REPORT
========================================
Generated by: House Application by Sanjana
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

PROPERTY DETAILS ANALYZED:
--------------------------
• Median Income: {st.session_state['predictions']['inputs']['Median Income']}
• House Age: {st.session_state['predictions']['inputs']['House Age']}
• Average Rooms: {st.session_state['predictions']['inputs']['Average Rooms']}
• Average Bedrooms: {st.session_state['predictions']['inputs']['Average Bedrooms']}
• Block Population: {st.session_state['predictions']['inputs']['Population']}
• Average Occupancy: {st.session_state['predictions']['inputs']['Average Occupancy']}
• Location: {st.session_state['predictions']['inputs']['Latitude']}, {st.session_state['predictions']['inputs']['Longitude']}

PRICE PREDICTIONS:
------------------
🔵 Linear Regression Model: {st.session_state['predictions']['predictions']['Linear Regression']}
🟣 Ridge Regression Model: {st.session_state['predictions']['predictions']['Ridge Regression']}
🟠 Lasso Regression Model: {st.session_state['predictions']['predictions']['Lasso Regression']}

🎯 RECOMMENDED MARKET VALUE: {st.session_state['predictions']['predictions']['Average Prediction']}

MODEL INFORMATION:
------------------
• Linear Regression: Basic linear relationship model
• Ridge Regression: Regularized model preventing overfitting
• Lasso Regression: Feature selection with L1 regularization
• Average: Mean of all three models for balanced prediction

DISCLAIMER:
-----------
These predictions are based on California housing data and machine learning models.
Actual property values may vary based on market conditions, specific property features,
and other factors not captured in this analysis.

For professional real estate advice, consult with licensed real estate professionals.

© 2024 House Application by Sanjana
"""
                
                st.success("📄 **Detailed prediction report ready for download!**")
                st.download_button(
                    label="📥 **Download Prediction Report**",
                    data=report_content,
                    file_name=f"House_Price_Prediction_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.warning("⚠️ Please generate a prediction first before downloading the report.")

with mainSection:
    show_main_page()



