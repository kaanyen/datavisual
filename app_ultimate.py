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

# Custom CSS for Professional UI
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #1f77b4;
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
    
    # Standardize Columns
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
        'Grade': 'Grade',
        'Nationality': 'Nationality'
    }
    df = df.rename(columns={k: v for k, v in cols_map.items() if k in df.columns})

    # Cleaning
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

    # Major Cleaning
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
    feat_cols = ['Gender', 'FinancialAid_Binary', 'Nationality_Grouped', 'Current_Major', 'Total_Courses', 'Has_Taken_Summer']
    model_df = student_agg.dropna(subset=feat_cols + ['CGPA', 'Is_Exited'])
    
    X = model_df[feat_cols]
    y_risk = model_df['AtRisk']
    y_gpa = model_df['CGPA']
    y_exit = model_df['Is_Exited']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['Total_Courses', 'Has_Taken_Summer', 'FinancialAid_Binary']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'Nationality_Grouped', 'Current_Major'])
    ])

    # 1. Risk Model (Logistic Regression)
    risk_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    risk_model.fit(X, y_risk)

    # 2. Dropout Model (Random Forest)
    exit_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])
    exit_model.fit(X, y_exit)

    # 3. GPA Model (Regressor)
    gpa_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    gpa_model.fit(X, y_gpa)

    # 4. Clustering (KMeans)
    X_unsup = student_agg[['CGPA', 'Total_Courses', 'Has_Taken_Summer']].copy().fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unsup)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(X_scaled)
    
    return risk_model, exit_model, gpa_model, kmeans, pca_res, clusters, scaler, X_scaled

# Load & Train
df, student_agg = load_data()
risk_model, exit_model, gpa_model, kmeans, pca_res, clusters, scaler, X_scaled = train_models(student_agg)
student_agg['Cluster'] = clusters
student_agg['PCA1'] = pca_res[:, 0]
student_agg['PCA2'] = pca_res[:, 1]
student_agg['Cluster_Label'] = student_agg['Cluster'].map({0: 'High Achiever (Summer)', 1: 'Standard High Achiever', 2: 'Remedial/Recovery'})

# =============================================================================
# MAIN APP LAYOUT
# =============================================================================
st.title("🎓 Ashesi Student Intelligence System")
st.markdown("Use the tabs below to navigate through the entire data story, explore patterns yourself, and simulate future outcomes.")

tabs = st.tabs([
    "📊 Executive Overview", 
    "🎯 Strategic Deep Dives", 
    "🕵️ Student Forensics", 
    "📉 Interactive Explorer", 
    "🔮 The Oracle (Predictions)", 
    "🎛️ Policy Simulator (What-If)"
])

# -----------------------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("University Pulse")
    
    # METRICS ROW
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{len(student_agg):,}</div><div class="metric-label">Total Students</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{student_agg["CGPA"].mean():.2f}</div><div class="metric-label">Average GPA</div></div>', unsafe_allow_html=True)
    retention = (1 - student_agg['Is_Exited'].mean()) * 100
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{retention:.1f}%</div><div class="metric-label">Retention Rate</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value">{student_agg["Current_Major"].nunique()}</div><div class="metric-label">Active Programs</div></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Admissions Growth")
        student_agg['Admit_Year_Clean'] = student_agg['Admission Year'].astype(str).str.split('-').str[0]
        admit_counts = student_agg['Admit_Year_Clean'].value_counts().sort_index().reset_index()
        admit_counts.columns = ['Year', 'Count']
        fig = px.bar(admit_counts, x='Year', y='Count', color='Count', title="New Student Admissions", color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Current Student Status")
        status_counts = student_agg['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig = px.pie(status_counts, names='Status', values='Count', hole=0.4, title="Status Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Diversity Profile")
    nat_counts = student_agg['Nationality_Grouped'].value_counts().reset_index()
    nat_counts.columns = ['Nationality', 'Count']
    fig = px.bar(nat_counts, x='Nationality', y='Count', color='Nationality', title="Top Nationalities")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: STRATEGIC DEEP DIVES
# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("Strategic Intelligence")
    
    # 1. CURRICULUM
    st.subheader("1. Curriculum Misalignment (Heatmap)")
    st.info("Showing failure rates by Major & Course. **Filtered for Non-Switchers** to show the true curriculum effect. Grey cells = Course not taken.")
    
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
    
    fig = px.imshow(misalign_heat, text_auto='.1%', aspect="auto", color_continuous_scale='Reds', title="Failure Rate Heatmap")
    st.plotly_chart(fig, use_container_width=True)

    # 2. MIGRATION
    st.subheader("2. Major Migration (Flow)")
    switchers = student_agg[student_agg['Did_Switch_Major'] == 1]
    if len(switchers) > 0:
        switch_counts = switchers.groupby(['Entry_Major', 'Current_Major']).size().reset_index(name='Count')
        switch_counts = switch_counts.sort_values('Count', ascending=False).head(10)
        
        fig = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = list(pd.concat([switch_counts['Entry_Major'], switch_counts['Current_Major']]).unique()),
              color = "blue"
            ),
            link = dict(
              source = [list(pd.concat([switch_counts['Entry_Major'], switch_counts['Current_Major']]).unique()).index(x) for x in switch_counts['Entry_Major']],
              target = [list(pd.concat([switch_counts['Entry_Major'], switch_counts['Current_Major']]).unique()).index(x) for x in switch_counts['Current_Major']],
              value = switch_counts['Count']
          ))])
        fig.update_layout(title_text="Top Major Switches (Entry -> Exit)", font_size=10)
        st.plotly_chart(fig, use_container_width=True)

    # 3. RISK TREND
    st.subheader("3. True Risk Trend")
    risk_trend = df[df['Sem_Num'] != 3].groupby(['Year', 'Semester'])['GPA'].apply(lambda x: (x < 2.0).mean() * 100).reset_index(name='Risk_Percent')
    risk_trend['Period'] = risk_trend['Year'] + ' ' + risk_trend['Semester']
    fig = px.line(risk_trend, x='Period', y='Risk_Percent', markers=True, title="Probation Rate (Excluding Summer)")
    fig.update_traces(line_color='red')
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: STUDENT FORENSICS
# -----------------------------------------------------------------------------
with tabs[2]:
    st.header("Ethical & Minority Intelligence")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Silent Departures")
        silent_leavers = student_agg[(student_agg['Status'] == 'Exited') & (student_agg['CGPA'] > 2.5)]
        st.error(f"{len(silent_leavers)} students with GPA > 2.5 dropped out.")
        st.dataframe(silent_leavers[['Gender', 'Nationality', 'CGPA', 'Current_Major']].head(10))
    
    with col2:
        st.subheader("Intersectionality Risk")
        intersect = student_agg.groupby(['Nationality_Grouped', 'FinancialAid_Binary'])['AtRisk'].mean().reset_index()
        intersect['Aid'] = intersect['FinancialAid_Binary'].map({0: 'No Aid', 1: 'Has Aid'})
        fig = px.density_heatmap(intersect, x='Nationality_Grouped', y='Aid', z='AtRisk', text_auto='.1%', title="Risk by Nation & Aid")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Linguistic Penalty Audit")
    writ_course = df[df['CourseName'].astype(str).str.contains("Written and Oral", case=False)]
    if not writ_course.empty:
        writ_code = writ_course['CourseCode'].iloc[0]
        writ_grades = df[df['CourseCode'] == writ_code]
        top_langs = writ_grades['NativeLanguage'].value_counts().head(5).index
        writ_grades_top = writ_grades[writ_grades['NativeLanguage'].isin(top_langs)]
        
        avg_grades = writ_grades_top.groupby('NativeLanguage')['Mark'].mean().reset_index()
        fig = px.bar(avg_grades, x='NativeLanguage', y='Mark', title="Avg Mark in 'Written Communication' by Language", color='Mark')
        fig.update_yaxes(range=[60, 80])
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: INTERACTIVE EXPLORER
# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("📉 Interactive Data Explorer")
    st.markdown("Build your own charts to find hidden patterns.")
    
    colA, colB, colC, colD = st.columns(4)
    x_axis = colA.selectbox("X Axis", options=student_agg.columns, index=0)
    y_axis = colB.selectbox("Y Axis", options=student_agg.select_dtypes(include=np.number).columns, index=0)
    color_var = colC.selectbox("Color By", options=['None'] + list(student_agg.columns), index=0)
    chart_type = colD.selectbox("Chart Type", ["Scatter", "Box", "Bar", "Histogram"])
    
    if chart_type == "Scatter":
        fig = px.scatter(student_agg, x=x_axis, y=y_axis, color=None if color_var == 'None' else color_var, hover_data=['CGPA', 'Current_Major'])
    elif chart_type == "Box":
        fig = px.box(student_agg, x=x_axis, y=y_axis, color=None if color_var == 'None' else color_var)
    elif chart_type == "Bar":
        fig = px.bar(student_agg, x=x_axis, y=y_axis, color=None if color_var == 'None' else color_var)
    elif chart_type == "Histogram":
        fig = px.histogram(student_agg, x=x_axis, color=None if color_var == 'None' else color_var)
        
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: THE ORACLE (PREDICTIONS)
# -----------------------------------------------------------------------------
with tabs[4]:
    st.header("🔮 Student Success Predictor")
    
    col_input, col_pred = st.columns([1, 2])
    
    with col_input:
        st.subheader("Student Profile")
        gender = st.selectbox("Gender", ["M", "F"])
        major = st.selectbox("Major", ["Business Admin", "Computer Science", "Engineering", "MIS", "Other"])
        courses = st.slider("Courses Completed", 0, 50, 10)
        summer = st.checkbox("Summer School?")
        finaid = st.checkbox("Financial Aid?")
        nation = st.selectbox("Nationality", ["Country0", "Country3", "Other"])
        
        input_data = pd.DataFrame({
            'Gender': [gender],
            'FinancialAid_Binary': [1 if finaid else 0],
            'Nationality_Grouped': [nation],
            'Current_Major': [major],
            'Total_Courses': [courses],
            'Has_Taken_Summer': [1 if summer else 0]
        })
        
        # Add missing cols for pipeline
        input_data['Entry_Major'] = major
        input_data['Did_Switch_Major'] = 0

    with col_pred:
        try:
            # Predict
            pred_risk = risk_model.predict_proba(input_data)[0][1]
            pred_gpa = gpa_model.predict(input_data)[0]
            pred_exit = exit_model.predict_proba(input_data)[0][1]
            
            # Cluster
            X_clust = pd.DataFrame({'CGPA': [pred_gpa], 'Total_Courses': [courses], 'Has_Taken_Summer': [1 if summer else 0]})
            X_clust_scaled = scaler.transform(X_clust)
            pred_cluster = kmeans.predict(X_clust_scaled)[0]
            
            # Display
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted GPA", f"{pred_gpa:.2f}")
            c2.metric("Dropout Probability", f"{pred_exit:.1%}")
            c3.metric("Behavioral Cluster", f"{pred_cluster}")
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = pred_risk * 100,
                title = {'text': "Risk Probability"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "red" if pred_risk > 0.5 else "green"}}
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            if pred_exit > 0.3: st.error("⚠️ HIGH DROPOUT RISK")
            elif pred_risk > 0.5: st.warning("⚠️ ACADEMIC PROBATION RISK")
            else: st.success("✅ ON TRACK")
            
        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------------------------------------------------------------
# TAB 6: POLICY SIMULATOR (WHAT-IF)
# -----------------------------------------------------------------------------
with tabs[5]:
    st.header("🎛️ Policy Lab: What-If Analysis")
    st.write("Simulate the impact of university-wide policy changes on average student outcomes.")
    
    colA, colB = st.columns(2)
    with colA:
        sim_finaid = st.slider("Target Financial Aid Coverage (%)", 0, 100, int(student_agg['FinancialAid_Binary'].mean()*100))
        sim_summer = st.slider("Target Summer Participation (%)", 0, 100, int(student_agg['Has_Taken_Summer'].mean()*100))
    
    if st.button("Run Simulation"):
        # Create Synthetic Cohort
        sim_df = student_agg.copy()
        
        # Adjust FinAid Distribution
        current_aid = sim_df['FinancialAid_Binary'].mean()
        target_aid = sim_finaid / 100.0
        # Randomly flip bits to meet target mean
        sim_df['FinancialAid_Binary'] = np.random.choice([0, 1], size=len(sim_df), p=[1-target_aid, target_aid])
        
        # Adjust Summer Distribution
        target_summer = sim_summer / 100.0
        sim_df['Has_Taken_Summer'] = np.random.choice([0, 1], size=len(sim_df), p=[1-target_summer, target_summer])
        
        # Run Predictions
        cols_needed = ['Gender', 'FinancialAid_Binary', 'Nationality_Grouped', 'Current_Major', 'Total_Courses', 'Has_Taken_Summer']
        # We need dummy cols for pipeline
        sim_df['Entry_Major'] = sim_df['Current_Major'] 
        sim_df['Did_Switch_Major'] = 0
        
        sim_gpa = gpa_model.predict(sim_df[cols_needed + ['Entry_Major', 'Did_Switch_Major']])
        sim_risk = risk_model.predict_proba(sim_df[cols_needed + ['Entry_Major', 'Did_Switch_Major']])[:, 1]
        
        # Results
        c1, c2 = st.columns(2)
        base_gpa = student_agg['CGPA'].mean()
        new_gpa = sim_gpa.mean()
        
        c1.metric("Projected Avg GPA", f"{new_gpa:.2f}", f"{new_gpa - base_gpa:.2f}")
        c2.metric("Projected Avg Risk", f"{sim_risk.mean():.1%}", f"{(sim_risk.mean() - student_agg['AtRisk'].mean())*100:.1f}%")
        
        fig = px.histogram(x=sim_gpa, nbins=30, title="Projected GPA Distribution", labels={'x': 'GPA'})
        fig.add_vline(x=base_gpa, line_dash="dash", line_color="red", annotation_text="Baseline")
        st.plotly_chart(fig, use_container_width=True)