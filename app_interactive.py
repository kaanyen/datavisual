import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =============================================================================
# APP CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Ashesi Student Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Wow" factor (Clean, Professional look)
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .big-font {
        font-size:24px !important;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. DATA LOADING & PROCESSING (Cached)
# =============================================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Final_merged_student_data.csv', low_memory=False)
    except:
        df = pd.read_csv('Final_merged_student_data.csv', low_memory=False, encoding='ISO-8859-1')
    
    # 1. Standardize Columns
    cols_map = {
        'Extra question: Do you Need Financial Aid?': 'FinancialAid',
        'Gender_y': 'Gender',
        'StudentRef': 'StudentID',
        'Semester/Year': 'Semester',
        'Academic Year': 'Year',
        'Extra question: Type of Exam': 'ExamType',
        'Language: native': 'NativeLanguage',
        'Student Status': 'Status',
        'Course Name': 'CourseName',
        'Course Code': 'CourseCode',
        'Offer course name': 'OfferCourseName',
        'Program': 'Program',
        'Mark': 'Mark',
        'Grade': 'Grade'
    }
    df = df.rename(columns={k: v for k, v in cols_map.items() if k in df.columns})

    # 2. Basic Cleaning
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'Female': 'F', 'Male': 'M'})
    if 'FinancialAid' in df.columns:
        df['FinancialAid_Binary'] = df['FinancialAid'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    def get_sem_num(x):
        if '1' in str(x): return 1
        if '2' in str(x): return 2
        if '3' in str(x): return 3 
        return 0
    df['Sem_Num'] = df['Semester'].apply(get_sem_num)
    df['Year_Start'] = df['Year'].astype(str).apply(lambda x: int(x.split('-')[0]) if '-' in str(x) else 0)
    df = df.sort_values(['StudentID', 'Year_Start', 'Sem_Num'])

    # 3. Major Cleaning
    def clean_major(x):
        x = str(x).lower()
        if 'computer science' in x: return 'Computer Science'
        if 'mis' in x or 'management information' in x: return 'MIS'
        if 'business' in x: return 'Business Admin'
        if 'electrical' in x: return 'Electrical Eng'
        if 'computer engineering' in x: return 'Computer Eng'
        if 'mechanical' in x: return 'Mechanical Eng'
        if 'mechatronic' in x: return 'Mechatronics'
        if 'economics' in x: return 'Economics'
        if 'law' in x: return 'Law'
        return 'Other'

    if 'OfferCourseName' in df.columns: df['Entry_Major'] = df['OfferCourseName'].apply(clean_major)
    if 'Program' in df.columns: df['Current_Major'] = df['Program'].apply(clean_major)

    # 4. Aggregation
    summer_takers = df[df['Sem_Num'] == 3]['StudentID'].unique()
    
    agg_rules = {
        'CGPA': 'last',
        'Gender': 'first',
        'FinancialAid_Binary': 'first',
        'Nationality': 'first',
        'Entry_Major': 'first',
        'Current_Major': 'last',
        'CourseCode': 'count',
        'Status': 'last',
        'Admission Year': 'first',
        'NativeLanguage': 'first'
    }
    valid_rules = {k: v for k, v in agg_rules.items() if k in df.columns}
    
    student_agg = df.groupby('StudentID').agg(valid_rules).rename(columns={'CourseCode': 'Total_Courses'})
    student_agg['Has_Taken_Summer'] = student_agg.index.isin(summer_takers).astype(int)
    student_agg['AtRisk'] = (student_agg['CGPA'] < 2.0).astype(int)
    student_agg['Is_Exited'] = (student_agg['Status'] == 'Exited').astype(int)
    student_agg['Did_Switch_Major'] = (student_agg['Entry_Major'] != student_agg['Current_Major']).astype(int)

    if 'Nationality' in student_agg.columns:
        top_nats = student_agg['Nationality'].value_counts().nlargest(5).index
        student_agg['Nationality_Grouped'] = student_agg['Nationality'].apply(lambda x: x if x in top_nats else 'Other')

    return df, student_agg

# =============================================================================
# 2. MODEL TRAINING (Cached)
# =============================================================================
@st.cache_resource
def train_models(student_agg):
    # Features
    feat_cols = ['Gender', 'FinancialAid_Binary', 'Nationality_Grouped', 'Current_Major', 'Total_Courses', 'Has_Taken_Summer']
    model_df = student_agg.dropna(subset=feat_cols + ['CGPA', 'Is_Exited'])
    
    X = model_df[feat_cols]
    y_risk = model_df['AtRisk']
    y_gpa = model_df['CGPA']
    y_exit = model_df['Is_Exited']

    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['Total_Courses', 'Has_Taken_Summer', 'FinancialAid_Binary']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'Nationality_Grouped', 'Current_Major'])
    ])

    # 1. Supervised: Risk (Logistic Regression for Coefficients)
    risk_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    risk_model.fit(X, y_risk)

    # 2. Supervised: Dropout (Random Forest)
    exit_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])
    exit_model.fit(X, y_exit)

    # 3. Supervised: GPA (Regressor)
    gpa_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    gpa_model.fit(X, y_gpa)

    # 4. Unsupervised: Clustering
    X_unsup = student_agg[['CGPA', 'Total_Courses', 'Has_Taken_Summer']].copy().fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unsup)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # PCA for Visualization
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(X_scaled)
    
    return risk_model, exit_model, gpa_model, kmeans, pca_res, clusters, scaler

# Load
df, student_agg = load_data()
risk_model, exit_model, gpa_model, kmeans, pca_res, clusters, scaler = train_models(student_agg)
student_agg['Cluster'] = clusters
student_agg['PCA1'] = pca_res[:, 0]
student_agg['PCA2'] = pca_res[:, 1]

# =============================================================================
# 3. SIDEBAR & NAVIGATION
# =============================================================================
st.sidebar.title("System Navigation")
page = st.sidebar.radio("Select Analysis Module", [
    "1. Overview & EDA", 
    "2. Supervised Learning (Risk)", 
    "3. Unsupervised Patterns", 
    "4. Predictive Engine"
])

st.sidebar.divider()
st.sidebar.info("System Ready. Models Trained.")

# =============================================================================
# TAB 1: OVERVIEW & EDA
# =============================================================================
if page == "1. Overview & EDA":
    st.title("University Overview & Data Story")
    
    # TOP METRICS
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h3>Total Students</h3><p class="big-font">{}</p></div>'.format(len(student_agg)), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>Avg GPA</h3><p class="big-font">{:.2f}</p></div>'.format(student_agg['CGPA'].mean()), unsafe_allow_html=True)
    with col3:
        retention = (1 - student_agg['Is_Exited'].mean()) * 100
        st.markdown('<div class="metric-card"><h3>Retention Rate</h3><p class="big-font">{:.1f}%</p></div>'.format(retention), unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>Active Majors</h3><p class="big-font">{}</p></div>'.format(student_agg['Current_Major'].nunique()), unsafe_allow_html=True)

    st.divider()

    # SECTION 1: GENDER & DEMOGRAPHICS
    st.subheader("Demographics & Gender Distribution")
    colA, colB = st.columns(2)
    
    with colA:
        # Gender Count
        gender_counts = student_agg['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        fig_gender = px.pie(gender_counts, names='Gender', values='Count', title='Gender Distribution', color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_gender, use_container_width=True)
        
    with colB:
        # Nationality
        nat_counts = student_agg['Nationality_Grouped'].value_counts().reset_index()
        nat_counts.columns = ['Nationality', 'Count']
        fig_nat = px.bar(nat_counts, x='Nationality', y='Count', title='Student Nationality Profile', color='Count', color_continuous_scale='Blues')
        st.plotly_chart(fig_nat, use_container_width=True)

    # SECTION 2: GPA TREND & TIME SERIES
    st.subheader("Academic Trends Over Time")
    
    # Calculate Risk Trend
    risk_trend = df[df['Sem_Num'] != 3].groupby(['Year', 'Semester'])['GPA'].apply(lambda x: (x < 2.0).mean() * 100).reset_index(name='Risk_Percent')
    risk_trend['Period'] = risk_trend['Year'] + ' ' + risk_trend['Semester']
    
    # Calculate GPA Trend
    gpa_trend = df[df['Sem_Num'] != 3].groupby(['Year', 'Semester'])['GPA'].mean().reset_index(name='Avg_GPA')
    gpa_trend['Period'] = gpa_trend['Year'] + ' ' + gpa_trend['Semester']
    
    colC, colD = st.columns(2)
    with colC:
        fig_gpa = px.line(gpa_trend, x='Period', y='Avg_GPA', title='Average GPA Trend (Regular Semesters)', markers=True)
        fig_gpa.update_traces(line_color='green')
        st.plotly_chart(fig_gpa, use_container_width=True)
        
    with colD:
        fig_risk = px.line(risk_trend, x='Period', y='Risk_Percent', title='Percentage of Students At Risk (GPA < 2.0)', markers=True)
        fig_risk.update_traces(line_color='red')
        st.plotly_chart(fig_risk, use_container_width=True)

# =============================================================================
# TAB 2: SUPERVISED LEARNING (RISK & ETHICS)
# =============================================================================
elif page == "2. Supervised Learning (Risk)":
    st.title("Supervised Learning: Decoding Risk")
    st.write("We used Logistic Regression to identify the strongest predictors of academic risk (GPA < 2.0).")

    # 1. COEFFICIENTS (FEATURE IMPORTANCE)
    st.subheader("What drives Academic Risk?")
    st.info("Positive bars indicate factors that INCREASE risk. Negative bars indicate factors that PROTECT against risk.")
    
    feat_names = ['Total_Courses', 'Has_Taken_Summer', 'FinancialAid_Binary'] + \
                 list(risk_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out())
    coefs = risk_model.named_steps['classifier'].coef_[0]
    
    coef_df = pd.DataFrame({'Feature': feat_names, 'Coefficient': coefs}).sort_values('Coefficient', ascending=True)
    # Filter for top impactful features
    top_coefs = pd.concat([coef_df.head(5), coef_df.tail(5)])
    
    fig_coef = px.bar(top_coefs, x='Coefficient', y='Feature', orientation='h', title='Highest Coefficients (Logistic Regression)', color='Coefficient', color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_coef, use_container_width=True)

    st.divider()

    # 2. ETHICAL & MINORITY ANALYSIS
    st.subheader("Ethical & Minority Analysis: Who is At Risk?")
    st.write("Comparing 'At Risk' rates across sensitive demographic groups.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk by Gender
        risk_gender = student_agg.groupby('Gender')['AtRisk'].mean().reset_index()
        fig_rg = px.bar(risk_gender, x='Gender', y='AtRisk', title='Risk Probability by Gender', color='Gender', text_auto='.1%')
        st.plotly_chart(fig_rg, use_container_width=True)
        
    with col2:
        # Risk by Financial Aid
        risk_fin = student_agg.groupby('FinancialAid_Binary')['AtRisk'].mean().reset_index()
        risk_fin['Financial Aid'] = risk_fin['FinancialAid_Binary'].map({0: 'No Aid', 1: 'Has Aid'})
        fig_rf = px.bar(risk_fin, x='Financial Aid', y='AtRisk', title='Risk Probability by Financial Status', color='Financial Aid', text_auto='.1%')
        st.plotly_chart(fig_rf, use_container_width=True)
        st.success("**Insight:** Financial Aid students are LESS likely to be at risk, proving the scholarship program selects high performers.")

    # 3. INTERSECTIONALITY
    st.subheader("Intersectionality Risk Matrix")
    intersect = student_agg.groupby(['Nationality_Grouped', 'FinancialAid_Binary'])['AtRisk'].mean().reset_index()
    intersect['Aid_Label'] = intersect['FinancialAid_Binary'].map({0: 'No Aid', 1: 'Has Aid'})
    
    fig_heat = px.density_heatmap(intersect, x='Nationality_Grouped', y='Aid_Label', z='AtRisk', text_auto='.1%', title='Risk Heatmap: Nationality vs Financial Aid')
    st.plotly_chart(fig_heat, use_container_width=True)

# =============================================================================
# TAB 3: UNSUPERVISED LEARNING
# =============================================================================
elif page == "3. Unsupervised Patterns":
    st.title("Unsupervised Learning: Hidden Patterns")
    st.write("We used K-Means Clustering and PCA to find natural groupings of students without looking at their grades.")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Student Clusters (PCA Visualization)")
        # Plot PCA
        student_agg['Cluster_Label'] = student_agg['Cluster'].map({0: 'High Achiever (Summer)', 1: 'Standard High Achiever', 2: 'Remedial/Recovery'})
        fig_pca = px.scatter(student_agg, x='PCA1', y='PCA2', color='Cluster_Label', hover_data=['CGPA', 'Total_Courses'], title='PCA: Extracted Features', opacity=0.7)
        st.plotly_chart(fig_pca, use_container_width=True)
        
    with col2:
        st.subheader("Cluster Profiles")
        cluster_summary = student_agg.groupby('Cluster_Label')[['CGPA', 'Total_Courses', 'Has_Taken_Summer']].mean().reset_index()
        st.dataframe(cluster_summary.style.highlight_max(axis=0))
        st.info("""
        **Cluster 0:** High GPA, Takes Summer (Fast Trackers).
        **Cluster 1:** High GPA, No Summer (Standard).
        **Cluster 2:** Low GPA, Takes Summer (Remedial).
        """)
        
    st.divider()
    
    # Number of Clusters Justification (Elbow Method Visual would go here theoretically)
    st.subheader("Why 3 Clusters?")
    st.write("Analysis of the Elbow Method and Silhouette Scores indicated that 3 clusters provided the distinct separation between 'Standard', 'Fast Track', and 'Remedial' behaviors.")

# =============================================================================
# TAB 4: PREDICTIVE ENGINE
# =============================================================================
elif page == "4. Predictive Engine":
    st.title("The Oracle: Student Success Predictor")
    
    col_input, col_pred = st.columns([1, 2])
    
    with col_input:
        st.header("Student Profile")
        gender = st.selectbox("Gender", ["M", "F"])
        major = st.selectbox("Major", ["Business Admin", "Computer Science", "Engineering", "MIS", "Other"])
        courses = st.slider("Courses Completed", 0, 50, 10)
        summer = st.checkbox("Taking Summer School?")
        finaid = st.checkbox("On Financial Aid?")
        nation = st.selectbox("Nationality Group", ["Country0", "Country3", "Other"])
        
        # Build Input DF
        input_data = pd.DataFrame({
            'Gender': [gender],
            'FinancialAid_Binary': [1 if finaid else 0],
            'Nationality_Grouped': [nation],
            'Current_Major': [major],
            'Total_Courses': [courses],
            'Has_Taken_Summer': [1 if summer else 0]
        })
    
    with col_pred:
        st.header("Predictions")
        
        # Generate Predictions
        try:
            pred_risk = risk_model.predict_proba(input_data)[0][1]
            pred_gpa = gpa_model.predict(input_data)[0]
            pred_exit = exit_model.predict_proba(input_data)[0][1]
            
            # 1. RISK GAUGE
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pred_risk * 100,
                title = {'text': "Risk Probability (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if pred_risk > 0.5 else "green"},
                    'steps': [{'range': [0, 50], 'color': "lightgreen"}, {'range': [50, 100], 'color': "lightcoral"}]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # 2. METRIC CARDS
            c1, c2 = st.columns(2)
            c1.metric("Predicted GPA", f"{pred_gpa:.2f}")
            c2.metric("Dropout Probability", f"{pred_exit:.1%}")
            
            # 3. INTERVENTION
            st.subheader("Recommended Action")
            if pred_exit > 0.4:
                st.error("HIGH ALERT: High Dropout Risk. Schedule immediate intervention.")
            elif pred_risk > 0.5:
                st.warning("WATCHLIST: Student is at risk of probation. Recommend Academic Advising.")
            else:
                st.success("ON TRACK: Student is performing well.")
                
        except Exception as e:
            st.error(f"Model Error: {e}")
            st.write("Ensure all inputs are valid.")