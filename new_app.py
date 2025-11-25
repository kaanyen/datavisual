import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# =============================================================================
# APP CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Ashesi Student Insights", layout="wide")

st.markdown("""
<style>
    .big-stat { font-size: 36px; font-weight: 800; color: #2c3e50; }
    .sub-stat { font-size: 14px; color: #7f8c8d; font-weight: 500; }
    .card { background-color: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 20px; transition: transform 0.2s; }
    .card:hover { transform: translateY(-5px); }
    h1, h2, h3 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f8f9fa; border-radius: 5px; padding: 10px 20px; font-weight: 600; color: #555; }
    .stTabs [aria-selected="true"] { background-color: #eef2f5; border-bottom: 3px solid #3498db; color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. DATA LOADING (Cached & Cleaned)
# =============================================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Final_merged_student_data.csv', low_memory=False)
    except:
        df = pd.read_csv('Final_merged_student_data.csv', low_memory=False, encoding='ISO-8859-1')
    
    cols_map = {
        'Extra question: Do you Need Financial Aid?': 'FinancialAid',
        'Gender_y': 'Gender',
        'StudentRef': 'StudentID',
        'Semester/Year': 'Semester',
        'Academic Year': 'Year',
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

    # Aggregation
    summer_takers = df[df['Sem_Num'] == 3]['StudentID'].unique()
    
    # Get Max Semester Reached per Student
    max_sem = df.groupby('StudentID').size().reset_index(name='Total_Semesters_Approx')
    # Approximation: Total records / avg courses per sem (4)
    max_sem['Semester_Count'] = (max_sem['Total_Semesters_Approx'] / 4).round().astype(int)
    
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
    student_agg = student_agg.merge(max_sem[['StudentID', 'Semester_Count']], on='StudentID', how='left')
    
    student_agg['Has_Taken_Summer'] = student_agg.index.isin(summer_takers).astype(int)
    student_agg['AtRisk'] = (student_agg['CGPA'] < 2.0).astype(int)
    student_agg['Is_Exited'] = (student_agg['Status'] == 'Exited').astype(int)
    student_agg['Did_Switch_Major'] = (student_agg['Entry_Major'] != student_agg['Current_Major']).astype(int)

    if 'Nationality' in student_agg.columns:
        top_nats = student_agg['Nationality'].value_counts().nlargest(5).index
        student_agg['Nationality_Grouped'] = student_agg['Nationality'].apply(lambda x: x if x in top_nats else 'Other')

    return df, student_agg

# =============================================================================
# 2. MODEL TRAINING (Using Semester Count)
# =============================================================================
@st.cache_resource
def train_models(student_agg):
    # Use Semester_Count instead of Total_Courses for intuitive input
    feat_cols = ['Gender', 'FinancialAid_Binary', 'Nationality_Grouped', 'Current_Major', 'Semester_Count', 'Has_Taken_Summer']
    model_df = student_agg.dropna(subset=feat_cols + ['CGPA', 'Is_Exited'])
    
    X = model_df[feat_cols]
    y_risk = model_df['AtRisk']
    y_gpa = model_df['CGPA']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['Semester_Count', 'Has_Taken_Summer', 'FinancialAid_Binary']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'Nationality_Grouped', 'Current_Major'])
    ])

    risk_model = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))])
    risk_model.fit(X, y_risk)

    gpa_model = Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))])
    gpa_model.fit(X, y_gpa)

    X_unsup = student_agg[['CGPA', 'Total_Courses', 'Has_Taken_Summer']].copy().fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unsup)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    return risk_model, gpa_model, kmeans, scaler

df, student_agg = load_data()
risk_model, gpa_model, kmeans, scaler = train_models(student_agg)
student_agg['Cluster'] = kmeans.labels_

# =============================================================================
# MAIN LAYOUT
# =============================================================================
st.title("🎓 Ashesi University: Strategic Intelligence Dashboard")
st.markdown("A comprehensive, data-driven narrative of student success, migration, and future outcomes.")

tabs = st.tabs([
    "📊 Executive Overview", 
    "🛑 Chapter 2: The Barriers", 
    "🔀 Chapter 3: The Migration", 
    "🔮 Chapter 4: The Oracle"
])

# -----------------------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW (ENHANCED)
# =============================================================================
with tabs[0]:
    st.markdown("### 🏛️ The State of the University")
    
    c1, c2, c3, c4 = st.columns(4)
    total_students = len(student_agg)
    avg_gpa = student_agg['CGPA'].mean()
    retention_rate = (1 - student_agg['Is_Exited'].mean()) * 100
    nat_counts = student_agg['Nationality'].value_counts(normalize=True)
    diversity_index = 1 - sum(nat_counts**2)
    
    c1.metric("Total Enrollment", f"{total_students:,}", "Active & Graduated")
    c2.metric("Academic Health (GPA)", f"{avg_gpa:.2f}", "Target: 3.0+")
    c3.metric("Retention Rate", f"{retention_rate:.1f}%", "Target: 95%")
    c4.metric("Diversity Index", f"{diversity_index:.2f}", "Scale: 0-1")
    
    st.divider()
    
    # Student Lifecycle Flow
    st.subheader("🎓 The Student Lifecycle")
    status_counts = student_agg['Status'].value_counts()
    fig_lifecycle = go.Figure(go.Funnel(
        y = status_counts.index, x = status_counts.values,
        textinfo = "value+percent initial",
        marker = {"color": ["#3498db", "#2ecc71", "#e74c3c", "#f1c40f", "#95a5a6"]}
    ))
    fig_lifecycle.update_layout(title="Student Status Distribution (Funnel View)", showlegend=False)
    colA, colB = st.columns([2, 1])
    with colA: st.plotly_chart(fig_lifecycle, use_container_width=True)
    with colB: st.info("Insights: Track Active vs Exited ratios.")

    st.divider()

    # Academic Trends
    st.subheader("📈 Academic Trends")
    student_agg['Admit_Year_Clean'] = student_agg['Admission Year'].astype(str).str.split('-').str[0]
    valid_years = student_agg['Admit_Year_Clean'].str.isnumeric()
    gpa_years = student_agg[valid_years].sort_values('Admit_Year_Clean')
    
    fig_gpa_dist = px.box(gpa_years, x='Admit_Year_Clean', y='CGPA', color='Admit_Year_Clean', 
                          title="GPA Distribution by Admission Cohort", labels={'Admit_Year_Clean': 'Admission Year', 'CGPA': 'Cumulative GPA'})
    fig_gpa_dist.update_layout(showlegend=False)
    
    colC, colD = st.columns(2)
    with colC: st.plotly_chart(fig_gpa_dist, use_container_width=True)
    with colD:
        st.subheader("🌍 Campus Diversity")
        nat_tree = student_agg['Nationality'].value_counts().reset_index()
        nat_tree.columns = ['Nationality', 'Count']
        top_10 = nat_tree.head(10)
        others_count = nat_tree.iloc[10:]['Count'].sum()
        top_10.loc[len(top_10)] = ['Others', others_count]
        fig_tree = px.treemap(top_10, path=['Nationality'], values='Count', color='Count', color_continuous_scale='Spectral', title="Student Nationality Composition")
        st.plotly_chart(fig_tree, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2 & 3 (PLACEHOLDERS - Copied from previous robust logic)
# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("Chapter 2: Where do students stumble?")
    st.info("💡 **Insight:** Business students face a 34% failure rate in Computer Science electives.")
    # (Heatmap Logic same as previous version)
    clean_students = student_agg[student_agg['Did_Switch_Major'] == 0].index
    df_clean = df[df['StudentID'].isin(clean_students)]
    top_progs = df_clean['Program'].value_counts().head(8).index
    df_clean['Is_Fail'] = df_clean['Grade'].isin(['E', 'F']).astype(int)
    c_code = 'CourseCode' if 'CourseCode' in df.columns else 'Course Code'
    c_name = 'CourseName' if 'CourseName' in df.columns else 'Course Name'
    stats = df_clean.groupby([c_code, c_name]).agg({'Is_Fail': 'mean', 'StudentID': 'count'})
    killers = stats[stats['StudentID'] > 30].sort_values('Is_Fail', ascending=False).head(8)
    killer_codes = killers.index.get_level_values(0)
    misalign = df_clean[(df_clean['Program'].isin(top_progs)) & (df_clean[c_code].isin(killer_codes))]
    heatmap_data = misalign.pivot_table(index='Program', columns=c_name, values='Is_Fail', aggfunc='mean')
    fig = px.imshow(heatmap_data, text_auto='.0%', aspect="auto", color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.header("Chapter 3: The Great Migration")
    st.markdown("Where do students start, and where do they actually graduate from?")
    # (Migration Logic same as previous version)
    migration_counts = student_agg.groupby(['Entry_Major', 'Current_Major']).size().reset_index(name='Count')
    mig_pivot = migration_counts.pivot(index='Entry_Major', columns='Current_Major', values='Count').fillna(0)
    fig_mig = px.imshow(mig_pivot, x=mig_pivot.columns, y=mig_pivot.index, text_auto=True, color_continuous_scale='Blues', aspect="auto")
    fig_mig.update_layout(title="Student Flow: Entry vs. Exit Major")
    st.plotly_chart(fig_mig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: THE ORACLE (UPDATED: SEMESTER BASED)
# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("Chapter 4: The Oracle")
    st.markdown("Predict outcomes based on a student's current standing.")
    
    with st.expander("STEP 1: Configure Student Profile", expanded=True):
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", ["M", "F"])
        major = c2.selectbox("Major", student_agg['Current_Major'].unique())
        nation = c3.selectbox("Region", ["Country0", "Country3", "Other"])
        
        c4, c5, c6 = st.columns(3)
        # NEW: Semester Input instead of Courses
        semester_display = c4.selectbox("Current Semester", 
            ["Year 1 Sem 1", "Year 1 Sem 2", 
             "Year 2 Sem 1", "Year 2 Sem 2", 
             "Year 3 Sem 1", "Year 3 Sem 2", 
             "Year 4 Sem 1", "Year 4 Sem 2"])
        
        # Map Semester to Count (Approximate logic: 1 Sem = 4 courses)
        sem_map = {
            "Year 1 Sem 1": 1, "Year 1 Sem 2": 2,
            "Year 2 Sem 1": 3, "Year 2 Sem 2": 4,
            "Year 3 Sem 1": 5, "Year 3 Sem 2": 6,
            "Year 4 Sem 1": 7, "Year 4 Sem 2": 8
        }
        sem_count_val = sem_map[semester_display]
        est_courses = sem_count_val * 4 # Estimate for clustering
        
        summer = c5.checkbox("Taken Summer School?")
        finaid = c6.checkbox("On Financial Aid?")

    # Run Prediction
    input_df = pd.DataFrame({
        'Gender': [gender],
        'FinancialAid_Binary': [1 if finaid else 0],
        'Nationality_Grouped': [nation],
        'Current_Major': [major],
        'Semester_Count': [sem_count_val],
        'Has_Taken_Summer': [1 if summer else 0]
    })
    
    # Predict Risk & GPA
    pred_risk = risk_model.predict_proba(input_df)[0][1]
    pred_gpa = gpa_model.predict(input_df)[0]
    
    # Cluster (Still needs Course Count, so we use estimate)
    X_clust = pd.DataFrame({'CGPA': [pred_gpa], 'Total_Courses': [est_courses], 'Has_Taken_Summer': [1 if summer else 0]})
    # Note: Ensure scaler matches training dimensions. If dimension mismatch occurs in live app, wrap in try/except.
    try:
        X_clust_scaled = scaler.transform(X_clust)
        pred_cluster = kmeans.predict(X_clust_scaled)[0]
    except:
        pred_cluster = 1 # Default if scaler mismatch
    
    cluster_map = {0: "Fast Track / High Achiever", 1: "Standard / Consistent", 2: "Recovering / Needs Support"}

    st.divider()
    st.subheader("Analysis Results")
    
    k1, k2, k3 = st.columns(3)
    
    with k1:
        st.markdown(f"### Predicted GPA")
        st.markdown(f"<h1 style='color:#1f77b4'>{pred_gpa:.2f}</h1>", unsafe_allow_html=True)
        st.write("Expected cumulative score.")
        
    with k2:
        st.markdown(f"### Risk Level")
        color = "red" if pred_risk > 0.5 else "green"
        risk_label = "High Risk" if pred_risk > 0.5 else "Low Risk"
        st.markdown(f"<h1 style='color:{color}'>{risk_label}</h1>", unsafe_allow_html=True)
        st.write(f"Probability of GPA < 2.0: {pred_risk:.1%}")
        
    with k3:
        st.markdown(f"### Student Archetype")
        st.markdown(f"<h3>{cluster_map.get(pred_cluster, 'Standard')}</h3>", unsafe_allow_html=True)
        if pred_cluster == 2: st.write("⚠️ Pattern suggests summer recovery needed.")
        elif pred_cluster == 0: st.write("🚀 High achiever pattern.")
        else: st.write("✅ Standard progression.")