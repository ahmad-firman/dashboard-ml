import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from prediction import StudentDropoutPredictor

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Student Dropout Prediction Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")


# Initialize predictor
@st.cache_resource
def get_predictor():
    return StudentDropoutPredictor()


# Title
st.markdown('<h1 class="main-header">🎓 Student Dropout Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox("Choose a page:", [
    "📈 Model Performance",
    "🔍 Feature Analysis",
    "👥 Student Risk Profiling",
    "🎯 Prediction Tool"
])

# Data for visualization (based on your analysis)
feature_importance = {
    'Feature': [
        'Second_sem_success_rate', 'Curricular_units_2nd_sem_approved',
        'First_sem_success_rate', 'Parent_education_gap', 'Min_parent_education',
        'Curricular_units_2nd_sem_enrolled', 'GDP', 'Unemployment_rate',
        'Displaced', 'Inflation_rate'
    ],
    'Importance': [0.275520, 0.112367, 0.041799, 0.039615, 0.034371,
                   0.029669, 0.029109, 0.026632, 0.025197, 0.023273]
}

risk_segmentation = {
    'Risk_Segment': ['Low Risk', 'Medium Risk', 'High Risk'],
    'Dropout_Rate': [8.5, 21.6, 66.6]
}

categorical_factors = {
    'Factor': ['Tuition_fees_up_to_date', 'Scholarship_holder', 'Debtor', 'Gender', 'Marital_status'],
    'Cramers_V': [0.428, 0.245, 0.229, 0.203, 0.116],
    'Max_Dropout_Rate': [86.6, 38.7, 62.0, 45.1, 66.7]
}

numerical_factors = {
    'Factor': ['Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_approved',
               'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_approved',
               'Age_at_enrollment'],
    'Cohens_d': [1.492, 1.483, 1.174, 1.169, 0.563],
    'Difference_Percent': [52.0, 65.5, 40.7, 55.4, 18.8]
}

# Model performance data
confusion_matrix_data = np.array([[109, 19], [8, 15]])
model_metrics = {
    'Metric': ['F1-Score', 'Precision', 'Recall', 'AUC-ROC'],
    'Optimized_Score': [0.5263, 0.4412, 0.6522, 0.7758],
    'Default_Score': [0.4545, 0.4762, 0.4348, 0.7758]
}

# PAGE 1: Model Performance
if page == "📈 Model Performance":
    st.header("🎯 Model Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Key Metrics Comparison")

        # Metrics comparison chart
        df_metrics = pd.DataFrame(model_metrics)
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df_metrics['Metric']))
        width = 0.35

        bars1 = ax.bar(x - width / 2, df_metrics['Default_Score'], width,
                       label='Default Threshold (0.5)', color='lightblue', alpha=0.8)
        bars2 = ax.bar(x + width / 2, df_metrics['Optimized_Score'], width,
                       label='Optimized Threshold (0.2)', color='darkblue', alpha=0.8)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance: Default vs Optimized Threshold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_metrics['Metric'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        st.pyplot(fig)

    with col2:
        st.subheader("🎯 Confusion Matrix")

        # Confusion Matrix Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted: Continue', 'Predicted: Dropout'],
                    yticklabels=['Actual: Continue', 'Actual: Dropout'],
                    ax=ax)
        ax.set_title('Confusion Matrix (Optimized Threshold = 0.2)')
        st.pyplot(fig)

    # Performance Summary
    st.subheader("📋 Performance Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🎯 F1-Score", "0.526", "0.072")
    with col2:
        st.metric("🎯 Precision", "0.441", "-0.035")
    with col3:
        st.metric("🎯 Recall", "0.652", "0.217")
    with col4:
        st.metric("❌ Missed Dropouts", "8", "-5")

# PAGE 2: Feature Analysis
elif page == "🔍 Feature Analysis":
    st.header("🔍 Feature Importance & Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Top 10 Feature Importance")

        # Feature importance chart
        df_features = pd.DataFrame(feature_importance)
        fig, ax = plt.subplots(figsize=(10, 8))

        bars = ax.barh(df_features['Feature'], df_features['Importance'],
                       color=plt.cm.viridis(np.linspace(0, 1, len(df_features))))
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance (XGBM Model)')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{width:.3f}', ha='left', va='center')

        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("📊 Categorical Factors Impact")

        # Categorical factors chart
        df_cat = pd.DataFrame(categorical_factors)
        fig, ax1 = plt.subplots(figsize=(10, 8))

        color = 'tab:red'
        ax1.set_xlabel('Factors')
        ax1.set_ylabel('Max Dropout Rate (%)', color=color)
        bars1 = ax1.bar(df_cat['Factor'], df_cat['Max_Dropout_Rate'],
                        color=color, alpha=0.7)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticklabels(df_cat['Factor'], rotation=45, ha='right')

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel("Cramér's V", color=color)
        line = ax2.plot(df_cat['Factor'], df_cat['Cramers_V'],
                        color=color, marker='o', linewidth=2, markersize=8)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Categorical Factors: Dropout Rate vs Statistical Significance')
        plt.tight_layout()
        st.pyplot(fig)

    # Numerical factors analysis
    st.subheader("📈 Numerical Factors Analysis")
    df_num = pd.DataFrame(numerical_factors)

    col1, col2 = st.columns(2)

    with col1:
        # Cohen's d chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(df_num['Factor'], df_num['Cohens_d'],
                       color='lightcoral', alpha=0.8)
        ax.set_xlabel("Cohen's d (Effect Size)")
        ax.set_title("Numerical Factors: Effect Size Analysis")
        ax.grid(axis='x', alpha=0.3)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{width:.3f}', ha='left', va='center')

        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        # Difference percentage chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(df_num['Factor'], df_num['Difference_Percent'],
                       color='lightblue', alpha=0.8)
        ax.set_xlabel('Difference Percentage (%)')
        ax.set_title('Numerical Factors: Percentage Difference')
        ax.grid(axis='x', alpha=0.3)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                    f'{width:.1f}%', ha='left', va='center')

        plt.tight_layout()
        st.pyplot(fig)

# PAGE 3: Student Risk Profiling
elif page == "👥 Student Risk Profiling":
    st.header("👥 Student Risk Profiling Analysis")

    # Risk segmentation overview
    st.subheader("🎯 Risk Segmentation Overview")
    df_risk = pd.DataFrame(risk_segmentation)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="risk-low">🟢 Low Risk Students</p>', unsafe_allow_html=True)
        st.metric("Dropout Rate", "8.5%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="risk-medium">🟡 Medium Risk Students</p>', unsafe_allow_html=True)
        st.metric("Dropout Rate", "21.6%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="risk-high">🔴 High Risk Students</p>', unsafe_allow_html=True)
        st.metric("Dropout Rate", "66.6%")
        st.markdown('</div>', unsafe_allow_html=True)

    # Dropout rates bar chart
    st.subheader("📈 Dropout Rate by Risk Segment")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ca02c', '#ff7f0e', '#d62728']
    bars = ax.bar(df_risk['Risk_Segment'], df_risk['Dropout_Rate'],
                  color=colors, alpha=0.8)

    ax.set_ylabel('Dropout Rate (%)')
    ax.set_title('Dropout Rate by Risk Segment')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    st.pyplot(fig)

    # High-risk student profile
    st.subheader("🚨 High-Risk Student Profile Characteristics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📊 Key Demographic Characteristics:**")
        st.markdown("- **Age at Enrollment:** 26.5 years (vs 23.3 overall)")
        st.markdown("- **Gender Distribution:** 54.1% Female, 45.9% Male")
        st.markdown("- **Marital Status:** 82.7% Single")
        st.markdown("- **Attendance:** 81.7% Daytime")
        st.markdown("- **Scholarship:** Only 3.8% have scholarships")
        st.markdown("- **Financial Issues:** 29.2% are debtors")
        st.markdown("- **Tuition Status:** 34.7% behind on fees")

    with col2:
        st.markdown("**📈 Academic Performance Indicators:**")
        st.markdown("- **Previous Qualification Grade:** 128.9 (vs 132.6 overall)")
        st.markdown("- **Admission Grade:** 123.1 (vs 127.0 overall)")
        st.markdown("- **1st Semester Grade:** 6.8 (vs 10.6 overall)")
        st.markdown("- **2nd Semester Grade:** 5.8 (vs 10.2 overall)")
        st.markdown("- **Economic Context:** Higher unemployment rate (12.2% vs 11.6%)")

# PAGE 4: Prediction Tool
elif page == "🎯 Prediction Tool":
    st.header("🎯 Student Dropout Prediction Tool")

    # Initialize predictor
    predictor = get_predictor()

    # Check if model is loaded
    if predictor.model_loaded:
        st.success(f"✅ {predictor.status_message}")

        st.subheader("📝 Student Information Input")
        st.markdown("Please fill in the student information below to get dropout prediction:")

        # Create input form using the prediction class
        with st.form("prediction_form"):
            # Get input form from predictor
            student_data = predictor.create_input_form()

            # Submit button
            submitted = st.form_submit_button("🔮 Predict Dropout Risk", use_container_width=True)

            if submitted:
                # Make prediction using the predictor
                result = predictor.make_prediction(student_data)
                predictor.display_prediction_results(result)
    else:
        st.error(f"❌ {predictor.status_message}")
        st.info("Please ensure the model file 'model.pkl' is in the same directory as this script.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666666;'>
        <p>🎓 Student Dropout Prediction Dashboard | Built with Streamlit</p>
        <p>📊 Powered by Machine Learning Model</p>
    </div>
    """,
    unsafe_allow_html=True
)