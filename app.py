
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import os

st.set_page_config(
    page_title="Student Academic Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .stApp { background: linear-gradient(135deg, #0d0d1a 0%, #111128 50%, #0d0d1a 100%); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d0d1a 0%, #111128 100%) !important;
            border-right: 1px solid rgba(218,165,32,0.2);
        }
        [data-testid="stSidebar"] label {
            color: #DAA520 !important;
            font-weight: 600 !important;
            font-size: 0.82em !important;
            letter-spacing: 0.5px;
        }
        .main-header {
            background: linear-gradient(135deg, #8B0000 0%, #D2691E 50%, #DAA520 100%);
            padding: 35px 40px;
            border-radius: 20px;
            margin-bottom: 25px;
            box-shadow: 0 20px 60px rgba(139,0,0,0.4);
            text-align: center;
        }
        .main-header h1 { color: white; font-size: 1.9em; font-weight: 800; margin: 0; letter-spacing: 1px; }
        .main-header p { color: rgba(255,255,255,0.85); font-size: 0.88em; margin-top: 8px; }
        .stat-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(218,165,32,0.2);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
        }
        .stat-value { font-size: 1.9em; font-weight: 800; color: #DAA520; }
        .stat-label { font-size: 0.72em; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }
        .result-graduate {
            background: linear-gradient(135deg, rgba(26,74,26,0.95), rgba(45,122,45,0.95));
            border: 2px solid #4CAF50;
            border-radius: 20px; padding: 35px; text-align: center;
            box-shadow: 0 15px 40px rgba(76,175,80,0.25);
        }
        .result-enrolled {
            background: linear-gradient(135deg, rgba(74,48,0,0.95), rgba(122,82,0,0.95));
            border: 2px solid #DAA520;
            border-radius: 20px; padding: 35px; text-align: center;
            box-shadow: 0 15px 40px rgba(218,165,32,0.25);
        }
        .result-dropout {
            background: linear-gradient(135deg, rgba(74,0,0,0.95), rgba(139,0,0,0.95));
            border: 2px solid #FF4444;
            border-radius: 20px; padding: 35px; text-align: center;
            box-shadow: 0 15px 40px rgba(255,68,68,0.25);
        }
        .result-title { font-size: 2.2em; font-weight: 800; color: white; margin: 0; letter-spacing: 3px; }
        .result-sub { color: rgba(255,255,255,0.8); font-size: 0.95em; margin-top: 12px; line-height: 1.6; }
        .prob-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 14px; padding: 18px 12px; text-align: center;
        }
        .prob-class { font-size: 0.72em; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; margin-bottom: 6px; }
        .prob-value { font-size: 2.0em; font-weight: 800; color: white; }
        .prob-bar-bg { background: rgba(255,255,255,0.08); border-radius: 10px; height: 6px; margin-top: 10px; overflow: hidden; }
        .section-title {
            color: #DAA520; font-size: 1.0em; font-weight: 700;
            text-transform: uppercase; letter-spacing: 2px;
            border-bottom: 1px solid rgba(218,165,32,0.25);
            padding-bottom: 8px; margin: 20px 0 12px 0;
        }
        .welcome-card {
            background: rgba(255,255,255,0.03);
            border: 1px dashed rgba(218,165,32,0.35);
            border-radius: 20px; padding: 70px 40px; text-align: center;
        }
        .insight-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px; padding: 12px 16px; margin: 6px 0;
        }
        .insight-label { color: rgba(255,255,255,0.4); font-size: 0.7em; text-transform: uppercase; letter-spacing: 1.5px; }
        .insight-value { color: white; font-size: 0.95em; font-weight: 600; margin-top: 2px; }
        .stButton > button {
            background: linear-gradient(135deg, #8B0000, #D2691E, #DAA520) !important;
            color: white !important; font-weight: 700 !important;
            font-size: 1.0em !important; border: none !important;
            border-radius: 12px !important; padding: 14px 28px !important;
            width: 100% !important; letter-spacing: 1px !important;
            box-shadow: 0 8px 25px rgba(139,0,0,0.35) !important;
        }
        .sidebar-section {
            background: rgba(218,165,32,0.08);
            border-left: 3px solid #DAA520;
            padding: 7px 12px; border-radius: 0 8px 8px 0;
            margin: 14px 0 8px 0; color: #DAA520;
            font-weight: 700; font-size: 0.8em; letter-spacing: 1px; text-transform: uppercase;
        }
        .action-card {
            border-radius: 0 12px 12px 0;
            padding: 14px 18px; margin-top: 10px;
        }
        .footer {
            text-align: center; color: rgba(255,255,255,0.25);
            font-size: 0.75em; padding: 30px 20px 10px;
            border-top: 1px solid rgba(255,255,255,0.06); margin-top: 50px;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model        = joblib.load("stacking_model.pkl")
    scaler       = joblib.load("scaler.pkl")
    le           = joblib.load("label_encoder.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, le, feature_names

model, scaler, le, feature_names = load_models()

marital_map = {"Single":1,"Married":2,"Widower":3,"Divorced":4,"Facto Union":5,"Legally Separated":6}
application_mode_map = {
    "1st phase - General quota":1,"Ordinance No. 612/93":2,
    "1st phase - Special quota (Madeira Island)":5,"Holders of other higher courses":7,
    "Ordinance No. 854-B/99":10,"International student (over 23 years)":15,
    "1st phase - Special quota (Azores Island)":16,"2nd phase - General quota":17,
    "3rd phase - General quota":18,"Over 23 years old":27,"Transfer":42,
    "Change of course":43,"Technological specialization diploma holders":44,
    "Change of institution/course":51,"Short cycle diploma holders":53,
    "Change of institution/course (International)":57,
}
course_map = {
    "Biofuel Production Technologies":33,"Animation and Multimedia Design":171,
    "Social Service (evening attendance)":8014,"Agronomy":9003,
    "Communication Design":9070,"Veterinary Nursing":9085,
    "Informatics Engineering (closest to Computer Science)":9119,"Equiniculture":9130,
    "Management":9147,"Social Service":9238,"Tourism":9254,"Nursing":9500,
    "Oral Hygiene":9556,"Management (evening attendance)":9670,
    "Journalism and Communication":9773,"Basic Education":9853,
    "Advertising and Marketing Management":9991,
}
prev_qual_map = {
    "Secondary education":1,"Higher education - Bachelor's degree":2,
    "Higher education - Degree":3,"Higher education - Master's":4,
    "Higher education - Doctorate":5,"Frequency of higher education":6,
    "12th year of schooling - not completed":9,"11th year of schooling - not completed":10,
    "Other - 11th year of schooling":12,"10th year of schooling":14,
    "Basic education 3rd cycle":19,"Technological specialization course":39,
    "Higher education - degree (1st cycle)":40,"Professional higher technical course":42,
    "Higher education - master (2nd cycle)":43,
}
nationality_map = {
    "Portuguese":1,"German":2,"Spanish":6,"Italian":11,"Dutch":13,"English":14,
    "Lithuanian":17,"Angolan":21,"Cape Verdean":22,"Guinean":24,"Mozambican":25,
    "Santomean":26,"Turkish":32,"Brazilian":41,"Romanian":62,
    "Moldova (Republic of)":100,"Mexican":101,"Ukrainian":103,"Russian":105,
    "Cuban":108,"Colombian":109,
}
qualification_map = {
    "Secondary Education - 12th Year or Equivalent":1,"Higher Education - Bachelor's Degree":2,
    "Higher Education - Degree":3,"Higher Education - Master's":4,
    "Higher Education - Doctorate":5,"Frequency of Higher Education":6,
    "12th Year - Not Completed":9,"11th Year - Not Completed":10,
    "Other - 11th Year":12,"10th Year of Schooling":14,"Basic Education 3rd Cycle":19,
    "Technical-professional course":22,"9th Year - Not Completed":29,
    "8th year of schooling":30,"Unknown":34,"Can't read or write":35,
    "Basic education 1st cycle":37,"Basic Education 2nd Cycle":38,
    "Technological specialization course":39,"Higher education - degree (1st cycle)":40,
    "Professional higher technical course":42,"Higher Education - Master (2nd cycle)":43,
    "Higher Education - Doctorate (3rd cycle)":44,
}
occupation_map = {
    "Student":0,"Representatives/Directors/Managers":1,
    "Specialists in Intellectual and Scientific Activities":2,
    "Intermediate Level Technicians":3,"Administrative staff":4,
    "Personal Services and Safety Workers":5,
    "Farmers and Skilled Workers in Agriculture":6,
    "Skilled Workers in Industry and Construction":7,
    "Equipment and Machines Operators":8,"Unskilled Workers":9,
    "Armed Forces":10,"Other Situation":90,"Health professionals":122,
    "Teachers":123,"ICT Specialists":125,"Science and Engineering Technicians":131,
    "Health Technicians":132,"Office Workers and Secretaries":141,
    "Financial and Accounting Support Workers":143,"Personal service workers":151,
    "Sellers":152,"Personal care workers":153,"Skilled construction workers":171,
    "Cleaning workers":191,"Unskilled agricultural workers":192,
    "Unskilled workers in industry and transport":193,"Meal preparation assistants":194,
}

# ── Banner ──
st.markdown("""
    <div class="main-header">
        <h1>🎓 Student Academic Performance Predictor</h1>
        <p>Hybridized Machine Learning Approach &nbsp;·&nbsp; Stacking Ensemble
        &nbsp;·&nbsp; Random Forest + Logistic Regression + Decision Tree + XGBoost</p>
        <p style="font-size:0.78em; opacity:0.65; margin-top:4px;">
        Amaraegbu Divine &nbsp;·&nbsp; Department of Computer Science &nbsp;·&nbsp; 2025</p>
    </div>
""", unsafe_allow_html=True)

# ── Stats Row ──
c1,c2,c3,c4 = st.columns(4)
for col, val, label in zip(
    [c1,c2,c3,c4],
    ["4,424","36","4","3"],
    ["Training Records","Input Features","ML Models","Outcome Classes"]
):
    col.markdown(f'''<div class="stat-card">
        <div class="stat-value">{val}</div>
        <div class="stat-label">{label}</div>
    </div>''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar ──
st.sidebar.markdown("""
    <div style='text-align:center;padding:18px 0 5px;'>
        <div style='font-size:2.2em;'>📋</div>
        <div style='color:#DAA520;font-weight:800;font-size:1.05em;margin-top:4px;'>Student Profile</div>
        <div style='color:rgba(255,255,255,0.35);font-size:0.72em;margin-top:2px;'>Fill in details and click Predict</div>
    </div>
""", unsafe_allow_html=True)

def user_input():
    st.sidebar.markdown('<div class="sidebar-section">🎓 Academic Info</div>', unsafe_allow_html=True)
    prev_qual_label = st.sidebar.selectbox("Previous Qualification", list(prev_qual_map.keys()))
    prev_qual = prev_qual_map[prev_qual_label]
    prev_qual_grade = st.sidebar.number_input("Prev. Qualification Grade (max: 200)", 0.0, 200.0, 120.0)
    admission_grade = st.sidebar.number_input("Admission Grade (max: 200)", 0.0, 200.0, 127.0)
    course_label = st.sidebar.selectbox("Course", list(course_map.keys()),
        index=list(course_map.keys()).index("Informatics Engineering (closest to Computer Science)"))
    course = course_map[course_label]
    st.sidebar.caption("ℹ️ Select the course closest to your student's field of study.")
    app_mode_label = st.sidebar.selectbox("Application Mode", list(application_mode_map.keys()))
    application_mode = application_mode_map[app_mode_label]
    application_order = st.sidebar.number_input("Application Order (0-9)", 0, 9, 1)
    attendance = st.sidebar.selectbox("Attendance", ["Daytime","Evening"])
    attendance = 1 if attendance == "Daytime" else 0

    st.sidebar.markdown('<div class="sidebar-section">📊 1st Semester</div>', unsafe_allow_html=True)
    st.sidebar.info("**Courses Credited:** Credit from prior learning\n\n**Courses Enrolled:** Registered for\n\n**Courses Evaluated:** Sat for assessment\n\n**Courses Approved:** Successfully passed\n\n**Average Grade:** Mean grade (max: 20)\n\n**Without Evaluations:** Never assessed")
    cu1_credited = st.sidebar.number_input("Courses Credited",  0, 20,  0, key="cu1c")
    cu1_enrolled = st.sidebar.number_input("Courses Enrolled",  0, 26,  6, key="cu1e")
    cu1_evals    = st.sidebar.number_input("Courses Evaluated", 0, 45,  6, key="cu1ev")
    cu1_approved = st.sidebar.number_input("Courses Approved",  0, 26,  5, key="cu1a")
    cu1_grade    = st.sidebar.number_input("Average Grade (max: 20)", 0.0, 20.0, 12.0, key="cu1g")
    cu1_no_eval  = st.sidebar.number_input("Without Evaluations", 0, 12, 0, key="cu1n")

    st.sidebar.markdown('<div class="sidebar-section">📊 2nd Semester</div>', unsafe_allow_html=True)
    st.sidebar.info("**Courses Credited:** Credit from prior learning\n\n**Courses Enrolled:** Registered for\n\n**Courses Evaluated:** Sat for assessment\n\n**Courses Approved:** Successfully passed\n\n**Average Grade:** Mean grade (max: 20)\n\n**Without Evaluations:** Never assessed")
    cu2_credited = st.sidebar.number_input("Courses Credited",  0, 20,  0, key="cu2c")
    cu2_enrolled = st.sidebar.number_input("Courses Enrolled",  0, 23,  6, key="cu2e")
    cu2_evals    = st.sidebar.number_input("Courses Evaluated", 0, 45,  6, key="cu2ev")
    cu2_approved = st.sidebar.number_input("Courses Approved",  0, 20,  5, key="cu2a")
    cu2_grade    = st.sidebar.number_input("Average Grade (max: 20)", 0.0, 20.0, 12.0, key="cu2g")
    cu2_no_eval  = st.sidebar.number_input("Without Evaluations", 0, 12, 0, key="cu2n")

    st.sidebar.markdown('<div class="sidebar-section">👤 Demographics</div>', unsafe_allow_html=True)
    age = st.sidebar.number_input("Age at Enrollment", 17, 70, 20)
    gender = st.sidebar.selectbox("Gender", ["Male","Female"])
    gender = 1 if gender == "Male" else 0
    marital_label = st.sidebar.selectbox("Marital Status", list(marital_map.keys()))
    marital_status = marital_map[marital_label]
    nationality_label = st.sidebar.selectbox("Nationality", list(nationality_map.keys()))
    nationality = nationality_map[nationality_label]
    international = st.sidebar.selectbox("International Student", ["No","Yes"])
    international = 1 if international == "Yes" else 0
    displaced = st.sidebar.selectbox("Displaced", ["No","Yes"])
    displaced = 1 if displaced == "Yes" else 0
    special_needs = st.sidebar.selectbox("Educational Special Needs", ["No","Yes"])
    special_needs = 1 if special_needs == "Yes" else 0

    st.sidebar.markdown('<div class="sidebar-section">💰 Socioeconomic</div>', unsafe_allow_html=True)
    tuition = st.sidebar.selectbox("Tuition Fees Up to Date", ["Yes","No"])
    tuition = 1 if tuition == "Yes" else 0
    scholarship = st.sidebar.selectbox("Scholarship Holder", ["No","Yes"])
    scholarship = 1 if scholarship == "Yes" else 0
    debtor = st.sidebar.selectbox("Debtor", ["No","Yes"])
    debtor = 1 if debtor == "Yes" else 0
    mothers_qual_label = st.sidebar.selectbox("Mother's Qualification", list(qualification_map.keys()))
    mothers_qual = qualification_map[mothers_qual_label]
    fathers_qual_label = st.sidebar.selectbox("Father's Qualification", list(qualification_map.keys()))
    fathers_qual = qualification_map[fathers_qual_label]
    mothers_occ_label = st.sidebar.selectbox("Mother's Occupation", list(occupation_map.keys()))
    mothers_occ = occupation_map[mothers_occ_label]
    fathers_occ_label = st.sidebar.selectbox("Father's Occupation", list(occupation_map.keys()))
    fathers_occ = occupation_map[fathers_occ_label]

    st.sidebar.markdown('<div class="sidebar-section">🌍 Macroeconomic</div>', unsafe_allow_html=True)
    unemployment = st.sidebar.number_input("Unemployment Rate (%)", 0.0, 25.0, 10.8)
    inflation    = st.sidebar.number_input("Inflation Rate (%)", -1.0, 5.0, 1.4)
    gdp          = st.sidebar.number_input("GDP", -5.0, 5.0, 1.74)

    data = [marital_status, application_mode, application_order, course,
            attendance, prev_qual, prev_qual_grade, nationality,
            mothers_qual, fathers_qual, mothers_occ, fathers_occ,
            admission_grade, displaced, special_needs, debtor,
            tuition, gender, scholarship, age, international,
            cu1_credited, cu1_enrolled, cu1_evals, cu1_approved,
            cu1_grade, cu1_no_eval, cu2_credited, cu2_enrolled,
            cu2_evals, cu2_approved, cu2_grade, cu2_no_eval,
            unemployment, inflation, gdp]

    profile = {
        "course": course_label, "age": age,
        "tuition": "Yes" if tuition == 1 else "No",
        "scholarship": "Yes" if scholarship == 1 else "No",
        "debtor": "Yes" if debtor == 1 else "No",
        "cu2_approved": cu2_approved, "cu2_grade": cu2_grade,
        "cu1_approved": cu1_approved, "cu1_grade": cu1_grade,
    }
    return np.array(data).reshape(1,-1), profile

input_data, profile = user_input()
predict_clicked = st.sidebar.button("🔍  PREDICT OUTCOME")

if predict_clicked:
    input_scaled   = scaler.transform(input_data)
    prediction     = model.predict(input_scaled)
    probabilities  = model.predict_proba(input_scaled)[0]
    predicted_class = le.inverse_transform(prediction)[0]
    class_probs    = dict(zip(le.classes_, probabilities))

    left_col, right_col = st.columns([3, 2])

    with left_col:
        if predicted_class == "Graduate":
            st.markdown('''<div class="result-graduate">
                <div style="font-size:3em;margin-bottom:12px;">✅</div>
                <div class="result-title">GRADUATE</div>
                <div class="result-sub">This student is predicted to successfully complete<br>
                their academic program. Continue monitoring<br>progress and celebrate achievements.</div>
            </div>''', unsafe_allow_html=True)
        elif predicted_class == "Enrolled":
            st.markdown('''<div class="result-enrolled">
                <div style="font-size:3em;margin-bottom:12px;">⚠️</div>
                <div class="result-title">ENROLLED</div>
                <div class="result-sub">This student is predicted to remain enrolled but<br>
                has not yet shown a strong graduation trajectory.<br>
                Academic engagement and support is recommended.</div>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown('''<div class="result-dropout">
                <div style="font-size:3em;margin-bottom:12px;">🚨</div>
                <div class="result-title">DROPOUT</div>
                <div class="result-sub">This student is at high risk of dropping out.<br>
                Immediate counseling, financial support, and<br>
                targeted intervention is strongly recommended.</div>
            </div>''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction Probabilities</div>', unsafe_allow_html=True)

        p1, p2, p3 = st.columns(3)
        dp = class_probs.get("Dropout", 0)
        ep = class_probs.get("Enrolled", 0)
        gp = class_probs.get("Graduate", 0)

        for col, cls, prob, color in zip(
            [p1,p2,p3],
            ["Dropout","Enrolled","Graduate"],
            [dp, ep, gp],
            ["#FF4444","#DAA520","#4CAF50"]
        ):
            col.markdown(f'''<div class="prob-card">
                <div class="prob-class" style="color:{color};">{cls}</div>
                <div class="prob-value">{prob:.1%}</div>
                <div class="prob-bar-bg">
                    <div style="background:{color};height:6px;border-radius:10px;width:{prob*100:.1f}%;"></div>
                </div>
            </div>''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(8,4))
        fig.patch.set_facecolor("#0d0d1a")
        ax.set_facecolor("#16213e")
        classes = list(le.classes_)
        probs_list = [class_probs.get(c,0) for c in classes]
        colors_list = ["#FF4444","#DAA520","#4CAF50"]
        bars = ax.bar(classes, probs_list, color=colors_list,
                      edgecolor="white", linewidth=0.4, width=0.45)
        ax.set_ylim(0,1)
        ax.set_ylabel("Probability", color="white", fontsize=11)
        ax.set_title("Class Probability Distribution", color="#DAA520", fontweight="bold", fontsize=13, pad=12)
        ax.tick_params(colors="white", labelsize=11)
        ax.bar_label(bars, fmt="%.3f", padding=5, color="white", fontweight="bold", fontsize=11)
        for spine in ax.spines.values(): spine.set_edgecolor("#333")
        ax.yaxis.grid(True, alpha=0.1, color="white")
        ax.set_axisbelow(True)
        st.pyplot(fig)
        plt.close()

    with right_col:
        st.markdown('<div class="section-title">Student Profile Summary</div>', unsafe_allow_html=True)
        course_display = profile["course"][:38]+"..." if len(profile["course"])>38 else profile["course"]
        for label, value in [
            ("Course", course_display),
            ("Age at Enrollment", str(profile["age"])),
            ("Tuition Fees Up to Date", profile["tuition"]),
            ("Scholarship Holder", profile["scholarship"]),
            ("Debtor Status", profile["debtor"]),
            ("2nd Sem Courses Approved", str(profile["cu2_approved"])),
            ("2nd Sem Average Grade", f"{profile['cu2_grade']:.1f} / 20"),
            ("1st Sem Courses Approved", str(profile["cu1_approved"])),
            ("1st Sem Average Grade", f"{profile['cu1_grade']:.1f} / 20"),
        ]:
            st.markdown(f'''<div class="insight-card">
                <div class="insight-label">{label}</div>
                <div class="insight-value">{value}</div>
            </div>''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recommended Action</div>', unsafe_allow_html=True)

        if predicted_class == "Graduate":
            ac, at = "#4CAF50", "Continue monitoring academic progress. Recognize and celebrate student achievements. Ensure graduation requirements are on track."
        elif predicted_class == "Enrolled":
            ac, at = "#DAA520", "Schedule an academic advising session. Review course engagement and attendance patterns. Connect the student with peer mentoring or tutoring resources."
        else:
            ac, at = "#FF4444", "URGENT: Contact the student immediately. Review available financial support options. Schedule a counseling session and assess academic workload and personal challenges."

        st.markdown(f'''<div style="background:rgba(255,255,255,0.03);border-left:4px solid {ac};
                    border-radius:0 12px 12px 0;padding:14px 18px;">
            <div style="color:{ac};font-weight:700;font-size:0.75em;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:8px;">Action Required</div>
            <div style="color:rgba(255,255,255,0.78);font-size:0.88em;line-height:1.6;">{at}</div>
        </div>''', unsafe_allow_html=True)

else:
    st.markdown("""
        <div class="welcome-card">
            <div style="font-size:3.5em;margin-bottom:18px;">🎓</div>
            <div style="color:#DAA520;font-size:1.4em;font-weight:700;margin-bottom:14px;">Ready to Predict</div>
            <div style="color:rgba(255,255,255,0.55);font-size:0.92em;line-height:1.7;max-width:480px;margin:0 auto;">
                Fill in the student details in the sidebar on the left, then click
                <strong style="color:#DAA520;">Predict Outcome</strong> to generate
                an AI-powered academic performance prediction.<br><br>
                The system uses a hybridized stacking ensemble of four machine learning models
                to deliver highly accurate predictions across three outcome classes:
                <strong style="color:#4CAF50;">Graduate</strong>,
                <strong style="color:#DAA520;">Enrolled</strong>, and
                <strong style="color:#FF4444;">Dropout</strong>.
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">📈 Model Performance & Insights</div>', unsafe_allow_html=True)

st.markdown("#### Model Performance Comparison")
if os.path.exists("model_comparison.png"):
    c1,c2,c3 = st.columns([1,3,1])
    with c2: st.image("model_comparison.png", use_container_width=True)
else:
    st.warning("Run the training notebook to generate model_comparison.png")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### Confusion Matrix — Stacking Ensemble")
if os.path.exists("confusion_matrix.png"):
    c1,c2,c3 = st.columns([1,3,1])
    with c2: st.image("confusion_matrix.png", use_container_width=True)
else:
    st.warning("Run the training notebook to generate confusion_matrix.png")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### Top 15 Feature Importances")
if os.path.exists("feature_importance.png"):
    c1,c2,c3 = st.columns([0.5,4,0.5])
    with c2: st.image("feature_importance.png", use_container_width=True)
else:
    st.warning("Run the training notebook to generate feature_importance.png")

st.markdown("""
    <div class="footer">
        Hybridized Machine Learning Approach for Student Academic Performance Prediction System<br>
        Amaraegbu Divine &nbsp;·&nbsp; Department of Computer Science &nbsp;·&nbsp; 2025<br><br>
        Stacking Ensemble: Random Forest + Logistic Regression + Decision Tree + XGBoost Meta-Learner
    </div>
""", unsafe_allow_html=True)
