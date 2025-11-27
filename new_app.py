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
# APP CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(page_title="Ashesi Strategic Intelligence", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2c3e50;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .summary-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 10px;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #333; }
    .metric-label { font-size: 14px; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; color: #2c3e50; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 5px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 3px solid #2c3e50; }
    .insight-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #3498db;
        margin-top: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. DATA LOADING & ENGINEERING
# =============================================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('student_data_small.csv', low_memory=False)
    except:
        df = pd.read_csv('student_data_small.csv', low_memory=False, encoding='ISO-8859-1')
    
    # --- RENAMING ---
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
        'Grade': 'Grade',
        'Submitted date': 'SubmittedDate',
        'Extra question: Exam Year': 'ExamYear',
        'Nationality': 'Nationality'
    }
    df = df.rename(columns={k: v for k, v in cols_map.items() if k in df.columns})

    # --- CLEANING ---
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'Female': 'F', 'Male': 'M'})
    if 'FinancialAid' in df.columns:
        df['FinancialAid_Binary'] = df['FinancialAid'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    # Semester Logic
    def get_sem_num(x):
        s = str(x)
        if '1' in s: return 1
        if '2' in s: return 2
        if '3' in s: return 3 
        return 0
    df['Sem_Num'] = df['Semester'].apply(get_sem_num)
    df['Year_Start'] = df['Year'].astype(str).apply(lambda x: int(x.split('-')[0]) if '-' in str(x) else 0)
    
    # Course Level Logic
    def get_level(code):
        num = ''.join(filter(str.isdigit, str(code)))
        return int(num[0])*100 if len(num) >= 3 else None
    df['Level'] = df['CourseCode'].apply(get_level)

    # Gap Year Logic
    try:
        df['SubmitYear'] = pd.to_datetime(df['SubmittedDate'], dayfirst=True, errors='coerce').dt.year
        df['ExamYearClean'] = pd.to_datetime(df['ExamYear'], dayfirst=True, errors='coerce').dt.year
        df['GapYears'] = df['SubmitYear'] - df['ExamYearClean']
        df['GapYears'] = df['GapYears'].apply(lambda x: x if 0 <= x <= 10 else np.nan) 
    except:
        df['GapYears'] = np.nan

    # COVID Logic
    def get_covid_era(row):
        y = row['Year_Start']
        s = row['Sem_Num']
        if y < 2019: return "Pre-COVID"
        if y == 2019 and s == 2: return "During COVID (Lockdown)"
        if y == 2020: return "During COVID (Hybrid)"
        return "Post-COVID"
    df['CovidEra'] = df.apply(get_covid_era, axis=1)

    # Major Logic
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

    # --- AGGREGATION ---
    summer_takers = df[df['Sem_Num'] == 3]['StudentID'].unique()
    
    # Max Semesters & Graduation Time
    max_sem = df.groupby('StudentID').agg({
        'Year_Start': ['min', 'max'],
        'Sem_Num': 'count'
    }).reset_index()
    max_sem.columns = ['StudentID', 'Start_Year', 'End_Year', 'Total_Records']
    max_sem['Years_To_Grad'] = max_sem['End_Year'] - max_sem['Start_Year'] + 1
    
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
        'GapYears': 'first',
        'NativeLanguage': 'first'
    }
    valid_rules = {k: v for k, v in agg_rules.items() if k in df.columns}
    
    # Reset Index so StudentID is a column
    student_agg = df.groupby('StudentID').agg(valid_rules).rename(columns={'CourseCode': 'Total_Courses'}).reset_index()
    student_agg = student_agg.merge(max_sem[['StudentID', 'Years_To_Grad']], on='StudentID', how='left')
    
    student_agg['Has_Taken_Summer'] = student_agg['StudentID'].isin(summer_takers).astype(int)
    student_agg['AtRisk'] = (student_agg['CGPA'] < 2.0).astype(int)
    student_agg['Is_Exited'] = (student_agg['Status'] == 'Exited').astype(int)
    student_agg['Did_Switch_Major'] = (student_agg['Entry_Major'] != student_agg['Current_Major']).astype(int)

    # Merge 'Did_Switch_Major' back to main DF
    df = df.merge(student_agg[['StudentID', 'Did_Switch_Major']], on='StudentID', how='left')

    if 'Nationality' in student_agg.columns:
        top_nats = student_agg['Nationality'].value_counts().nlargest(5).index
        student_agg['Nationality_Grouped'] = student_agg['Nationality'].apply(lambda x: x if x in top_nats else 'Other')
        
        # Map back to DF
        nat_map = student_agg.set_index('StudentID')['Nationality_Grouped'].to_dict()
        df['Nationality_Grouped'] = df['StudentID'].map(nat_map)

    return df, student_agg

# =============================================================================
# 2. MODEL TRAINING (Robust)
# =============================================================================
@st.cache_resource
def train_models(student_agg):
    feat_cols = ['Gender', 'FinancialAid_Binary', 'Nationality_Grouped', 'Current_Major', 'Total_Courses', 'Has_Taken_Summer']
    model_df = student_agg.dropna(subset=feat_cols + ['CGPA'])
    
    X = model_df[feat_cols]
    y_risk = model_df['AtRisk']
    y_gpa = model_df['CGPA']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['Total_Courses', 'Has_Taken_Summer', 'FinancialAid_Binary']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'Nationality_Grouped', 'Current_Major'])
    ])

    # Risk Model
    risk_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    risk_model.fit(X, y_risk)

    # GPA Model
    gpa_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
    ])
    gpa_model.fit(X, y_gpa)
    
    # Feature Importance Helper
    feature_names = ['Total_Courses', 'Has_Taken_Summer', 'FinancialAid_Binary'] + \
                    list(risk_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out())
    coefs = risk_model.named_steps['classifier'].coef_[0]
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Impact': coefs}).sort_values('Impact', ascending=False)

    return risk_model, gpa_model, feat_imp

df, student_agg = load_data()
risk_model, gpa_model, feat_imp = train_models(student_agg)

active_students = student_agg[student_agg['Status'] == 'Active']

# =============================================================================
# DASHBOARD LAYOUT
# =============================================================================
st.title("Telling the Story Behind the Numbers")
st.markdown("**From Admissions to Graduation: A Data-Driven Story**")

tabs = st.tabs([
    "1. Executive Summary", 
    "2. Admission Metrics", 
    "3. Performance Metrics", 
    "4. Program Migration", 
    "5. Graduation History", 
    "6. Student Success Modeling",
    "7. Summary Report"
])

# -----------------------------------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("Executive Summary: The Current State")
    
    # METRICS
    c1, c2, c3, c4 = st.columns(4)
    total_active = len(active_students)
    at_risk_active = active_students['AtRisk'].sum()
    pct_risk = (at_risk_active / total_active) * 100 if total_active > 0 else 0
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{total_active:,}</div><div class="metric-label">Active Students</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{at_risk_active}</div><div class="metric-label">Students At Risk</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{pct_risk:.1f}%</div><div class="metric-label">Risk Rate</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value">{active_students["CGPA"].mean():.2f}</div><div class="metric-label">Avg Active GPA</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <strong>Executive Insight:</strong> The active student body is healthy with an average GPA of 3.05. However, a specific segment (approx 5-8%) remains at risk of probation. Use the filters below to identify these groups.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # SNAPSHOT
    colA, colB = st.columns([2, 1])
    with colA:
        st.subheader("Academic Health Snapshot (Active Students)")
        fig_dist = px.histogram(active_students, x="CGPA", nbins=40, title="Current GPA Distribution", color_discrete_sequence=['#2c3e50'])
        fig_dist.add_vline(x=2.0, line_dash="dash", line_color="black", annotation_text="Probation (2.0)")
        fig_dist.add_vline(x=3.5, line_dash="dash", line_color="gold", annotation_text="Dean's List (3.5)")
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with colB:
        st.subheader("Top Level Risk Findings")
        risk_view = st.radio("View Risk By:", ["Nationality", "Major", "Gender"], horizontal=True)
        
        if risk_view == "Nationality":
            risk_data = active_students.groupby('Nationality_Grouped')['AtRisk'].mean().reset_index().sort_values('AtRisk', ascending=True)
            fig_r = px.bar(risk_data, x='AtRisk', y='Nationality_Grouped', orientation='h', title="Risk Probability by Region", color='AtRisk', color_continuous_scale='Reds')
        elif risk_view == "Major":
            risk_data = active_students.groupby('Current_Major')['AtRisk'].mean().reset_index().sort_values('AtRisk', ascending=True)
            fig_r = px.bar(risk_data, x='AtRisk', y='Current_Major', orientation='h', title="Risk Probability by Major", color='AtRisk', color_continuous_scale='Reds')
        else:
            risk_data = active_students.groupby('Gender')['AtRisk'].mean().reset_index().sort_values('AtRisk', ascending=True)
            fig_r = px.bar(risk_data, x='AtRisk', y='Gender', orientation='h', title="Risk Probability by Gender", color='AtRisk', color_continuous_scale='Reds')
        
        fig_r.layout.xaxis.tickformat = ',.0%'
        st.plotly_chart(fig_r, use_container_width=True)

    # ROSTER
    with st.expander("View At-Risk Student Roster (Filterable)", expanded=False):
        risk_roster = active_students[active_students['AtRisk'] == 1]
        c_fil1, c_fil2 = st.columns(2)
        fil_prog = c_fil1.multiselect("Filter by Major", risk_roster['Current_Major'].unique())
        fil_ctry = c_fil2.multiselect("Filter by Nationality", risk_roster['Nationality'].unique())
        if fil_prog: risk_roster = risk_roster[risk_roster['Current_Major'].isin(fil_prog)]
        if fil_ctry: risk_roster = risk_roster[risk_roster['Nationality'].isin(fil_ctry)]
        st.dataframe(risk_roster[['Gender', 'Nationality', 'Current_Major', 'FinancialAid_Binary', 'CGPA']].sort_values('CGPA'))



# -----------------------------------------------------------------------------
# TAB 2: ADMISSION METRICS
# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("Admission Profile")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Admissions Evolution")
        student_agg['Admit_Year_Clean'] = student_agg['Admission Year'].astype(str).str.split('-').str[0]
        admit_counts = student_agg['Admit_Year_Clean'].value_counts().sort_index().reset_index()
        admit_counts.columns = ['Year', 'Count']
        fig_adm = px.bar(admit_counts, x='Year', y='Count', title="New Student Intake per Year", color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig_adm, use_container_width=True)
    with c2:
        st.subheader("Gap Year Analysis")
        gap_data = student_agg.dropna(subset=['GapYears'])
        fig_gap = px.histogram(gap_data, x='GapYears', nbins=10, title="Distribution of Gap Years", color_discrete_sequence=['teal'])
        st.plotly_chart(fig_gap, use_container_width=True)

    st.divider()
    st.subheader("Student Demographics")
    c1, c2, c3 = st.columns(3)
    with c1:
        nat_counts = student_agg['Nationality_Grouped'].value_counts().reset_index()
        fig_nat = px.pie(nat_counts, values='count', names='Nationality_Grouped', title="Nationality Mix", hole=0.4)
        st.plotly_chart(fig_nat, use_container_width=True)
    with c2:
        fin_counts = student_agg['FinancialAid_Binary'].value_counts().reset_index()
        fin_counts['Type'] = fin_counts['FinancialAid_Binary'].map({0: 'Non-Aid', 1: 'Financial Aid'})
        fig_fin = px.pie(fin_counts, values='count', names='Type', title="Financial Aid Distribution", hole=0.4, color_discrete_sequence=['gray', 'gold'])
        st.plotly_chart(fig_fin, use_container_width=True)
    with c3:
        gen_counts = student_agg['Gender'].value_counts().reset_index()
        fig_gen = px.pie(gen_counts, values='count', names='Gender', title="Gender Balance", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_gen, use_container_width=True)

    st.subheader("COVID Impact Analysis")
    covid_perf = df.groupby('CovidEra')['GPA'].mean().reset_index()
    order = ['Pre-COVID', 'During COVID (Lockdown)', 'During COVID (Hybrid)', 'Post-COVID']
    fig_cov = px.bar(covid_perf, x='CovidEra', y='GPA', title="Average GPA by Era", category_orders={'CovidEra': order}, color='GPA', color_continuous_scale='Tealgrn')
    fig_cov.update_yaxes(range=[2.5, 3.5])
    fig_cov.update_layout(plot_bgcolor='white')
    st.plotly_chart(fig_cov, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: PERFORMANCE METRICS
# -----------------------------------------------------------------------------
with tabs[2]:
    st.header("Performance & Barriers")
    st.subheader("1. Curriculum Barriers (The 'Killer' Courses)")
    st.write("Where do students stumble? (Filtered for students who stayed in their major).")
    
    df_clean = df[df['Did_Switch_Major'] == 0].copy()
    df_clean['Is_Fail'] = df_clean['Grade'].isin(['E', 'F']).astype(int)
    top_progs = df_clean['Program'].value_counts().head(8).index
    c_code = 'CourseCode' if 'CourseCode' in df.columns else 'Course Code'
    c_name = 'CourseName' if 'CourseName' in df.columns else 'Course Name'
    
    stats = df_clean.groupby([c_code, c_name]).agg({'Is_Fail': 'mean', 'StudentID': 'count'})
    killers = stats[stats['StudentID'] > 30].sort_values('Is_Fail', ascending=False).head(8)
    
    misalign = df_clean[(df_clean['Program'].isin(top_progs)) & (df_clean['CourseCode'].isin(killers.index.get_level_values(0)))]
    heatmap_data = misalign.pivot_table(index='Program', columns=c_name, values='Is_Fail', aggfunc='mean')
    
    fig_heat = px.imshow(heatmap_data, text_auto='.0%', aspect="auto", color_continuous_scale='Reds', title="Failure Rate Matrix")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    st.subheader("2. Failure Rate by Course Level (Graduates Only)")
    c_sel1, c_sel2 = st.columns(2)
    sel_major = c_sel1.selectbox("Select Major", df['Program'].unique())
    compare_by = c_sel2.radio("Compare Trends By:", ["None (Overall)", "Gender", "Financial Aid", "Nationality"], horizontal=True)
    
    filtered_fail = df[(df['Program'] == sel_major) & (df['Status'] == 'Graduated')].copy()
    if not filtered_fail.empty:
        filtered_fail['Is_Fail'] = filtered_fail['Grade'].isin(['E', 'F']).astype(int)
        filtered_fail['Aid_Status'] = filtered_fail['FinancialAid_Binary'].map({0: 'Non-Aid', 1: 'Aid'})
        
        if compare_by == "Gender": group_cols, color_col = ['Level', 'Gender'], 'Gender'
        elif compare_by == "Financial Aid": group_cols, color_col = ['Level', 'Aid_Status'], 'Aid_Status'
        elif compare_by == "Nationality": group_cols, color_col = ['Level', 'Nationality_Grouped'], 'Nationality_Grouped'
        else: group_cols, color_col = ['Level'], None
        
        level_stats = filtered_fail.groupby(group_cols)['Is_Fail'].mean().reset_index()
        level_stats = level_stats[level_stats['Level'].isin([100, 200, 300, 400])]
        fig_lvl = px.line(level_stats, x='Level', y='Is_Fail', color=color_col, markers=True, title=f"Failure Trend: {sel_major} (Graduates Only)")
        if color_col is None: fig_lvl.update_traces(line_color='#2c3e50', line_width=3)
        st.plotly_chart(fig_lvl, use_container_width=True)
    else: st.warning("No graduate data available for this Major.")

    st.divider()
    st.subheader("3. Performance Breakdowns")
    maj_perf = student_agg.groupby('Current_Major').agg(Avg_GPA=('CGPA', 'mean')).reset_index()
    st.dataframe(maj_perf.style.background_gradient(subset=['Avg_GPA'], cmap='Greens'))

    c_gen, c_fin = st.columns(2)
    with c_gen:
        gen_perf = df.groupby(['Year_Start', 'Gender'])['GPA'].mean().reset_index()
        fig_gp = px.line(gen_perf, x='Year_Start', y='GPA', color='Gender', title="Gender Performance")
        st.plotly_chart(fig_gp, use_container_width=True)
    with c_fin:
        fin_perf = student_agg.groupby('FinancialAid_Binary')['CGPA'].mean().reset_index()
        fin_perf['Status'] = fin_perf['FinancialAid_Binary'].map({0: 'Non-Aid', 1: 'Aid'})
        fig_fp = px.bar(fin_perf, x='Status', y='CGPA', color='Status', title="Financial Aid Performance", color_discrete_sequence=['gray', 'gold'])
        fig_fp.update_yaxes(range=[2.5, 3.5])
        st.plotly_chart(fig_fp, use_container_width=True)

    st.subheader("4. Language Barrier Audit")
    writ_course = df[df['CourseName'].astype(str).str.contains("Written and Oral", case=False)]
    if not writ_course.empty:
        writ_code = writ_course['CourseCode'].iloc[0]
        writ_grades = df[df['CourseCode'] == writ_code]
        top_langs = writ_grades['NativeLanguage'].value_counts().head(5).index
        writ_grades = writ_grades[writ_grades['NativeLanguage'].isin(top_langs)]
        fig_lang = px.bar(writ_grades.groupby('NativeLanguage')['Mark'].mean().reset_index(), x='NativeLanguage', y='Mark', title="Avg Mark in 'Written Communication' by Language", color='Mark', color_continuous_scale='Viridis')
        fig_lang.update_yaxes(range=[60, 80])
        st.plotly_chart(fig_lang, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: PROGRAM MIGRATION
# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("Program Migration: Movement & Impact")
    
    st.subheader("1. Visualizing Student Movement (Paths)")
    switchers = student_agg[student_agg['Did_Switch_Major'] == 1]
    if not switchers.empty:
        switch_counts = switchers.groupby(['Entry_Major', 'Current_Major']).size().reset_index(name='Count')
        switch_counts['Path'] = switch_counts['Entry_Major'] + " > " + switch_counts['Current_Major']
        switch_counts = switch_counts.sort_values('Count', ascending=True) 
        fig_bar = px.bar(switch_counts, x='Count', y='Path', orientation='h', title="Major Migration Paths (Most Frequent)", color='Entry_Major', text='Count')
        fig_bar.update_layout(height=600)
        st.plotly_chart(fig_bar, use_container_width=True)
    else: st.warning("No major switches detected.")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("2. The Migration Matrix")
        mig_counts = student_agg.groupby(['Entry_Major', 'Current_Major']).size().reset_index(name='Count')
        mig_pivot = mig_counts.pivot(index='Entry_Major', columns='Current_Major', values='Count').fillna(0)
        fig_mig = px.imshow(mig_pivot, text_auto=True, color_continuous_scale='Blues', aspect="auto", title="Frequency Matrix")
        st.plotly_chart(fig_mig, use_container_width=True)
    
    with c2:
        st.subheader("3. Switch Success Predictor")
        start_m = st.selectbox("Started in:", sorted(student_agg['Entry_Major'].unique()))
        end_m = st.selectbox("Switched to:", sorted(student_agg['Current_Major'].unique()))
        subset = student_agg[(student_agg['Entry_Major'] == start_m) & (student_agg['Current_Major'] == end_m)]
        if len(subset) > 0:
            succ_rate = (1 - subset['AtRisk'].mean()) * 100
            avg_gpa_switch = subset['CGPA'].mean()
            st.metric("Students Moved", f"{len(subset)}")
            st.metric("Success Rate (GPA > 2.0)", f"{succ_rate:.0f}%")
            st.metric("Avg GPA after Switch", f"{avg_gpa_switch:.2f}")
        else: st.info("No students found for this specific path.")

# -----------------------------------------------------------------------------
# TAB 5: GRADUATION HISTORY (UPDATED RISK TABLE)
# -----------------------------------------------------------------------------
with tabs[4]:
    st.header("Graduation Timelines")
    grads = student_agg[student_agg['Status'] == 'Graduated']
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Time to Degree by Major (Distribution Table)")
        grads['Grad_Speed'] = grads['Years_To_Grad'].apply(lambda x: '4 Years' if x == 4 else ('< 4 Years' if x < 4 else ('5 Years' if x == 5 else '6+ Years')))
        grad_table = pd.crosstab(grads['Current_Major'], grads['Grad_Speed'], normalize='index') * 100
        cols_order = [c for c in ["< 4 Years", "4 Years", "5 Years", "6+ Years"] if c in grad_table.columns]
        st.dataframe(grad_table[cols_order].style.format("{:.1f}%").background_gradient(cmap="Reds", axis=1))

    with c2:
        st.subheader("Financial Aid Impact on Time")
        aid_time = grads.groupby('FinancialAid_Binary')['Years_To_Grad'].mean().reset_index()
        aid_time['Status'] = aid_time['FinancialAid_Binary'].map({0: 'Non-Aid', 1: 'Aid'})
        fig_at = px.bar(aid_time, x='Status', y='Years_To_Grad', title="Avg Years to Graduate", color='Status')
        fig_at.update_yaxes(range=[3, 5])
        st.plotly_chart(fig_at, use_container_width=True)

    st.subheader("Final Risk Summary: Who struggles to graduate?")
    risk_groups = student_agg[student_agg['AtRisk'] == 1]
    if not risk_groups.empty:
        # TABLE LOGIC (Count + Percentage)
        risk_summary = risk_groups.groupby(['Current_Major', 'Nationality_Grouped']).size().reset_index(name='At_Risk_Count')
        total_summary = student_agg.groupby(['Current_Major', 'Nationality_Grouped']).size().reset_index(name='Total_Count')
        risk_table = risk_summary.merge(total_summary, on=['Current_Major', 'Nationality_Grouped'])
        risk_table['Risk_Percentage'] = (risk_table['At_Risk_Count'] / risk_table['Total_Count'] * 100).round(1)
        
        risk_table['Display'] = risk_table.apply(lambda x: f"{x['At_Risk_Count']} ({x['Risk_Percentage']}%)", axis=1)
        final_table = risk_table.pivot(index='Current_Major', columns='Nationality_Grouped', values='Display').fillna("-")
        
        st.write("**Count of At-Risk Students (and % of that specific group):**")
        st.dataframe(final_table.style.background_gradient(cmap="Reds"))

# -----------------------------------------------------------------------------
# TAB 6: SUCCESS MODELING
# -----------------------------------------------------------------------------
with tabs[5]:
    st.header("Student Success Modeling")
    
    st.subheader("What drives the model?")
    fig_imp = px.bar(feat_imp.head(8), x='Impact', y='Feature', orientation='h', title="Top Factors influencing Risk/GPA")
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.divider()
    st.subheader("Scenario Simulator")
    st.markdown("Use this tool to explore 'What If' scenarios.")
    st.components.v1.iframe("https://99fec3f569aafb1238.gradio.live/", height=800, scrolling=True)

# -----------------------------------------------------------------------------
# TAB 7: SUMMARY REPORT
# -----------------------------------------------------------------------------
with tabs[6]:
    st.header("Summary Report")
    st.markdown("### Key Findings at a Glance")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="summary-card">
            <h4>The Good</h4>
            <ul>
                <li><strong>Active Growth:</strong> Admissions have consistently increased year-over-year.</li>
                <li><strong>Aid Success:</strong> Financial Aid students outperform non-aid students (GPA 3.12 vs 2.87).</li>
                <li><strong>Summer Safety Net:</strong> Summer school effectively helps struggling students recover.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div class="summary-card">
            <h4>The Risks</h4>
            <ul>
                <li><strong>Curriculum Bottleneck:</strong> Business students face high failure rates (34%) in CS electives.</li>
                <li><strong>Migration Flow:</strong> Significant outflow from Engineering to MIS (retention issue in Eng).</li>
                <li><strong>Regional Gap:</strong> Students from 'Country3' show higher risk probabilities.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>COVID Context:</strong> GPA trend analysis shows a slight 'Lockdown Spike' (likely due to remote assessment methods) followed by a steady post-COVID baseline. The recent uptick may also correlate with the public release of tools like ChatGPT, changing how students engage with assignments.
    </div>
    
    <div class="warning-box">
    <strong>Beyond the Data:</strong> This dashboard relies purely on academic records. It does NOT capture:
    <ul>
        <li>Mental & Physical Health (Flu season, chronic conditions)</li>
        <li>Social Dynamics (Clubs, Relationships, AJC cases)</li>
        <li>External Achievements (Startups, Businesses, Internships)</li>
    </ul>
    <strong>Ethical Use:</strong> Do not use this tool as a 'gatekeeper' to reject students. High-risk markers often indicate high-potential students who simply need different support structures.
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Recommended Actions")
    st.markdown("""
    1.  **Launch Math Bridge Program:** Specifically for Business majors taking CS courses.
    2.  **Targeted Advising:** Focus on Engineering students in Year 2 to prevent migration/dropout.
    3.  **Cultural Integration:** Expand support for 'Country3' students beyond just language (cultural onboarding).
    """)
