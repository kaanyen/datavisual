import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from math import pi

# PAGE CONFIGURATION
st.set_page_config(page_title="Ashesi Student Intelligence", layout="wide")

# =============================================================================
# 1. CACHED DATA LOADING & PROCESSING
# =============================================================================
@st.cache_data
def load_and_process_data():
    try:
        df = pd.read_csv('Final_merged_student_data.csv', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv('Final_merged_student_data.csv', low_memory=False, encoding='ISO-8859-1')

    # Standardize Column Names
    cols_map = {
        'Extra question: Do you Need Financial Aid?': 'FinancialAid',
        'Gender_y': 'Gender',
        'StudentRef': 'StudentID',
        'Semester/Year': 'Semester',
        'Academic Year': 'Year',
        'Extra question: Type of Exam': 'ExamType',
        'Extra question: How did you hear about Ashesi? You can select all that apply to you:': 'MarketingChannel',
        'Language: native': 'NativeLanguage',
        'Student Status': 'Status',
        'Course Name': 'CourseName',
        'Course Code': 'CourseCode',
        'Offer course name': 'OfferCourseName',
        'Mark': 'Mark',
        'Grade': 'Grade',
        'Program': 'Program'
    }
    df = df.rename(columns={k: v for k, v in cols_map.items() if k in df.columns})

    # Standardize Values
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'Female': 'F', 'Male': 'M'})
    if 'FinancialAid' in df.columns:
        df['FinancialAid_Binary'] = df['FinancialAid'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    # Parse Semester Logic
    def get_sem_num(x):
        s = str(x)
        if '1' in s: return 1
        if '2' in s: return 2
        if '3' in s: return 3 
        return 0
    df['Sem_Num'] = df['Semester'].apply(get_sem_num)
    df['Year_Start'] = df['Year'].astype(str).apply(lambda x: int(x.split('-')[0]) if '-' in str(x) else 0)
    df = df.sort_values(['StudentID', 'Year_Start', 'Sem_Num'])

    # Feature Engineering: Major Cleaning
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

    if 'OfferCourseName' in df.columns and 'Program' in df.columns:
        df['Entry_Major'] = df['OfferCourseName'].apply(clean_major)
        df['Current_Major'] = df['Program'].apply(clean_major)

    # Student-Level Aggregation
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

    # Derived Features
    student_agg['Has_Taken_Summer'] = student_agg.index.isin(summer_takers).astype(int)
    student_agg['AtRisk'] = (student_agg['CGPA'] < 2.0).astype(int)
    student_agg['Is_Exited'] = (student_agg['Status'] == 'Exited').astype(int)
    student_agg['Did_Switch_Major'] = (student_agg['Entry_Major'] != student_agg['Current_Major']).astype(int)

    # Group Rare Nationalities
    if 'Nationality' in student_agg.columns:
        top_nats = student_agg['Nationality'].value_counts().nlargest(5).index
        student_agg['Nationality_Grouped'] = student_agg['Nationality'].apply(lambda x: x if x in top_nats else 'Other')
    
    return df, student_agg

# =============================================================================
# 2. CACHED MODEL TRAINING
# =============================================================================
@st.cache_resource
def train_all_models(student_agg):
    # Prepare Data
    feat_cols = ['Gender', 'FinancialAid_Binary', 'Nationality_Grouped', 'Entry_Major', 'Current_Major', 'Total_Courses', 'Has_Taken_Summer', 'Did_Switch_Major']
    model_df = student_agg.dropna(subset=feat_cols + ['CGPA', 'Is_Exited'])

    X = model_df[feat_cols]
    y_risk = model_df['AtRisk']
    y_gpa = model_df['CGPA']
    y_exit = model_df['Is_Exited']

    # Preprocessing
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['Total_Courses', 'Has_Taken_Summer', 'FinancialAid_Binary', 'Did_Switch_Major']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'Nationality_Grouped', 'Entry_Major', 'Current_Major'])
    ])

    # Model 1: Risk Classifier
    risk_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    risk_model.fit(X, y_risk)

    # Model 2: GPA Regressor
    gpa_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
    ])
    gpa_model.fit(X, y_gpa)

    # Model 3: Dropout Classifier
    exit_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42))
    ])
    exit_model.fit(X, y_exit)
    
    # Unsupervised: KMeans
    X_unsup = student_agg[['CGPA', 'Total_Courses', 'Has_Taken_Summer']].copy().fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unsup)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)

    return risk_model, gpa_model, exit_model, kmeans, scaler

# Load Data
df, student_agg = load_and_process_data()
risk_model, gpa_model, exit_model, kmeans, scaler = train_all_models(student_agg)

# =============================================================================
# DASHBOARD LAYOUT
# =============================================================================

# SIDEBAR
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Executive Overview", "Strategic Deep Dives", "Student Forensics", "The Oracle (Predictions)"])

# ------------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# ------------------------------------------------------------------
if page == "Executive Overview":
    st.title("University Pulse: Executive Brief")
    st.markdown("### The Growth, The Diversity, and The Outcomes")
    
    # METRICS
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Enrollment", f"{len(student_agg):,}")
    col2.metric("Average CGPA", f"{student_agg['CGPA'].mean():.2f}")
    retention = (1 - student_agg['Is_Exited'].mean()) * 100
    col3.metric("Retention Rate", f"{retention:.1f}%")
    col4.metric("Active Majors", f"{student_agg['Current_Major'].nunique()}")
    
    st.divider()

    # CHARTS
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Admissions Growth Trend")
        student_agg['Admit_Year_Clean'] = student_agg['Admission Year'].astype(str).str.split('-').str[0]
        admit_counts = student_agg['Admit_Year_Clean'].value_counts().sort_index()
        fig, ax = plt.subplots()
        sns.barplot(x=admit_counts.index, y=admit_counts.values, palette="Blues_d", ax=ax)
        plt.xticks(rotation=45)
        plt.ylabel("New Students")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Student Status Distribution")
        status_counts = student_agg['Status'].value_counts()
        fig, ax = plt.subplots()
        plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"))
        st.pyplot(fig)
    
    st.subheader("Campus Diversity (Top Nationalities)")
    top_nats = student_agg['Nationality'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=top_nats.values, y=top_nats.index, palette="Spectral", ax=ax)
    plt.xlabel("Student Count")
    st.pyplot(fig)

# ------------------------------------------------------------------
# TAB 2: STRATEGIC DEEP DIVES
# ------------------------------------------------------------------
elif page == "Strategic Deep Dives":
    st.title("Strategic Intelligence")
    st.markdown("### Uncovering Hidden Patterns & Curriculum Bottlenecks")
    
    # 1. CURRICULUM HEATMAP
    st.subheader("1. Curriculum Misalignment")
    st.info("**Insight:** We filtered out major switchers to show the 'Pure' curriculum. Grey cells indicate courses that are not part of that major's typical path.")
    
    clean_students = student_agg[student_agg['Did_Switch_Major'] == 0].index
    df_clean = df[df['StudentID'].isin(clean_students)]
    
    top_10_progs = df_clean['Program'].value_counts().head(10).index
    df_clean['Is_Fail'] = df_clean['Grade'].isin(['E', 'F']).astype(int)
    
    code_col = 'CourseCode' if 'CourseCode' in df.columns else 'Course Code'
    name_col = 'CourseName' if 'CourseName' in df.columns else 'Course Name'
    
    course_stats = df_clean.groupby([code_col, name_col]).agg({'Is_Fail': 'mean', 'StudentID': 'count'})
    top_10_killer = course_stats[course_stats['StudentID'] > 30].sort_values('Is_Fail', ascending=False).head(10)
    killer_codes = top_10_killer.index.get_level_values(0)
    
    misalign = df_clean[(df_clean['Program'].isin(top_10_progs)) & (df_clean[code_col].isin(killer_codes))]
    misalign_heat = misalign.pivot_table(index='Program', columns=name_col, values='Is_Fail', aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    mask = misalign_heat.isnull()
    sns.heatmap(misalign_heat, annot=True, fmt=".1%", cmap="Reds", mask=mask, cbar_kws={'label': 'Failure Rate'}, vmin=0, vmax=0.15, ax=ax)
    ax.set_facecolor('lightgrey')
    plt.xlabel('')
    plt.ylabel('')
    st.pyplot(fig)
    
    # 2. MIGRATION
    st.subheader("2. The Major Migration")
    st.write("Tracking how students move between programs (Entry -> Graduation).")
    switchers = student_agg[student_agg['Did_Switch_Major'] == 1]
    if len(switchers) > 0:
        switch_counts = switchers.groupby(['Entry_Major', 'Current_Major']).size().reset_index(name='Count')
        switch_counts = switch_counts.sort_values('Count', ascending=False).head(10)
        switch_counts['Path'] = switch_counts['Entry_Major'] + " -> " + switch_counts['Current_Major']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=switch_counts, y='Path', x='Count', palette="rocket", ax=ax)
        plt.xlabel("Number of Switchers")
        st.pyplot(fig)
    
    # 3. TRUE RISK TREND
    st.subheader("3. True Risk Trend")
    st.write("Percentage of students on probation (GPA < 2.0) excluding Summer semesters.")
    risk_trend = df[df['Sem_Num'] != 3].groupby(['Year', 'Semester'])['GPA'].apply(lambda x: (x < 2.0).mean() * 100).reset_index(name='Risk_Percent')
    risk_trend['Period'] = risk_trend['Year'] + ' ' + risk_trend['Semester']
    
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=risk_trend, x='Period', y='Risk_Percent', marker='o', color='darkred', linewidth=2, ax=ax)
    plt.xticks(rotation=45)
    plt.ylabel("% At Risk")
    st.pyplot(fig)

# ------------------------------------------------------------------
# TAB 3: STUDENT FORENSICS
# ------------------------------------------------------------------
elif page == "Student Forensics":
    st.title("Ethical & Minority Intelligence")
    st.markdown("### Identifying Bias and Hidden Departures")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("The 'Silent Departures'")
        silent_leavers = student_agg[(student_agg['Status'] == 'Exited') & (student_agg['CGPA'] > 2.5)]
        st.error(f"{len(silent_leavers)} High-Performing Students (GPA > 2.5) left the university.")
        st.write("This suggests financial or cultural barriers, not academic ones.")
        st.dataframe(silent_leavers[['Gender', 'Nationality', 'CGPA', 'Current_Major']].head(10))
        
    with col2:
        st.subheader("Intersectionality Risk Matrix")
        st.write("Risk probability by Demographic & Financial Status.")
        intersect_risk = student_agg.groupby(['Gender', 'Nationality_Grouped', 'FinancialAid_Binary'])['AtRisk'].mean().reset_index()
        intersect_risk['Label'] = intersect_risk['Gender'] + " | " + intersect_risk['Nationality_Grouped']
        pivot = intersect_risk.pivot(index='Label', columns='FinancialAid_Binary', values='AtRisk')
        pivot.columns = ['No Aid', 'Has Aid']
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(pivot, annot=True, fmt=".1%", cmap="Reds", ax=ax)
        st.pyplot(fig)
        
    st.divider()
    
    st.subheader("Linguistic Penalty Audit")
    st.write("Does native language affect performance in 'Written and Oral Communication'?")
    writ_course = df[df['CourseName'].astype(str).str.contains("Written and Oral", case=False)]
    if not writ_course.empty:
        writ_code = writ_course['CourseCode'].iloc[0]
        writ_grades = df[df['CourseCode'] == writ_code]
        top_langs = writ_grades['NativeLanguage'].value_counts().head(5).index
        writ_grades_top = writ_grades[writ_grades['NativeLanguage'].isin(top_langs)]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=writ_grades_top, x='NativeLanguage', y='Mark', palette="viridis", errorbar=None, ax=ax)
        plt.ylim(60, 80)
        plt.ylabel("Average Mark")
        st.pyplot(fig)

# ------------------------------------------------------------------
# TAB 4: THE ORACLE (PREDICTIONS)
# ------------------------------------------------------------------
elif page == "The Oracle (Predictions)":
    st.title("Student Success Predictor")
    st.markdown("### Real-time AI Inference Engine")
    st.write("Input a student profile to generate 5 predictions using our trained Random Forest & Logistic Regression models.")
    
    # SIDEBAR INPUTS
    st.sidebar.header("Configure Profile")
    gender = st.sidebar.selectbox("Gender", ["M", "F"])
    finaid = st.sidebar.selectbox("Financial Aid?", ["No", "Yes"])
    nation = st.sidebar.selectbox("Region/Nationality", ["Country0", "Country3", "Other"])
    major_in = st.sidebar.selectbox("Entry Major", ["Computer Science", "Business Admin", "Engineering", "MIS"])
    major_now = st.sidebar.selectbox("Current Major", ["Computer Science", "Business Admin", "Engineering", "MIS"])
    courses = st.sidebar.slider("Courses Completed", 0, 50, 10)
    summer = st.sidebar.checkbox("Has Taken Summer School?")
    
    # Process Inputs
    input_data = pd.DataFrame({
        'Gender': [gender],
        'FinancialAid_Binary': [1 if finaid == 'Yes' else 0],
        'Nationality_Grouped': [nation],
        'Entry_Major': [major_in],
        'Current_Major': [major_now],
        'Total_Courses': [courses],
        'Has_Taken_Summer': [1 if summer else 0],
        'Did_Switch_Major': [1 if major_in != major_now else 0]
    })
    
    # INFERENCE
    pred_risk_prob = risk_model.predict_proba(input_data)[0][1]
    pred_gpa = gpa_model.predict(input_data)[0]
    pred_exit_prob = exit_model.predict_proba(input_data)[0][1]
    
    # Cluster Inference
    X_unsup_input = pd.DataFrame({
        'CGPA': [pred_gpa],
        'Total_Courses': [courses],
        'Has_Taken_Summer': [1 if summer else 0]
    })
    # Align columns for scaler
    # Note: Scaler expects 3 cols: CGPA, Total_Courses, Has_Taken_Summer
    X_scaled_input = scaler.transform(X_unsup_input)
    pred_cluster = kmeans.predict(X_scaled_input)[0]
    
    cluster_labels = {
        0: "High Achiever (Fast Track)",
        1: "Standard Performer",
        2: "Recovering / Remedial"
    }
    
    # INTERVENTION LOGIC
    if pred_exit_prob > 0.4:
        intervention = "High Priority: Immediate Intervention Required (Dropout Risk)"
    elif pred_risk_prob > 0.5:
        intervention = "Medium Priority: Schedule Academic Advising (Grade Risk)"
    elif pred_cluster == 2:
        intervention = "Watchlist: Monitor Summer Performance"
    else:
        intervention = "Low Priority: On Track"

    # DISPLAY RESULTS
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Risk Assessment")
        color = "red" if pred_risk_prob > 0.5 else "green"
        st.markdown(f"<h2 style='color:{color}'>{pred_risk_prob:.1%}</h2>", unsafe_allow_html=True)
        st.caption("Probability of GPA < 2.0")
        
    with col2:
        st.subheader("2. GPA Forecast")
        st.metric("Expected CGPA", f"{pred_gpa:.2f}")
        
    with col3:
        st.subheader("3. Dropout Sentinel")
        st.progress(min(pred_exit_prob, 1.0))
        st.caption(f"{pred_exit_prob:.1%} Probability of Exit")
        
    st.divider()
    
    col4, col5 = st.columns(2)
    with col4:
        st.subheader("4. Behavioral Profile")
        st.info(f"**Cluster {pred_cluster}:** {cluster_labels.get(pred_cluster, 'Unknown')}")
        
    with col5:
        st.subheader("5. AI Recommendation")
        if "High Priority" in intervention:
            st.error(intervention)
        elif "Medium Priority" in intervention:
            st.warning(intervention)
        else:
            st.success(intervention)