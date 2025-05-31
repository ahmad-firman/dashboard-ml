import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from typing import Dict, Tuple, Any
import xgboost


class StudentDropoutPredictor:
    """
    Class untuk handling prediksi student dropout
    """

    def __init__(self, model_path: str = "model.pkl"):
        """
        Initialize predictor dengan load model

        Args:
            model_path: Path ke file model pickle
        """
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        self.status_message = ""
        self._load_model()

    def _load_model(self):
        """Load model dari file pickle"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as file:
                    self.model = pickle.load(file)
                self.model_loaded = True
                self.status_message = "Model loaded successfully!"
            except Exception as e:
                self.model_loaded = False
                self.status_message = f"Error loading model: {str(e)}"
        else:
            self.model_loaded = False
            self.status_message = f"Model file '{self.model_path}' not found in the current directory"

    def create_input_form(self) -> Dict[str, Any]:
        """
        Membuat form input untuk data student dan return dictionary data

        Returns:
            Dictionary berisi data student
        """
        student_data = {}

        # Personal Information
        st.markdown("### 👤 Personal Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            student_data['marital_status'] = st.selectbox(
                "Marital Status",
                options=[1, 2, 3, 4, 5, 6],
                format_func=lambda x: {1: "Single", 2: "Married", 3: "Widower",
                                       4: "Divorced", 5: "Facto Union", 6: "Legally Separated"}[x],
                index=0
            )

        with col2:
            student_data['age_at_enrollment'] = st.number_input("Age at Enrollment",
                                                                min_value=17, max_value=70, value=20)

        with col3:
            student_data['gender'] = st.selectbox(
                "Gender",
                options=[0, 1],
                format_func=lambda x: "Male" if x == 0 else "Female",
                index=0
            )

        # Academic Background
        st.markdown("### 🎓 Academic Background")
        col1, col2, col3 = st.columns(3)

        with col1:
            student_data['application_mode'] = st.number_input("Application Mode", min_value=1, max_value=57, value=1)

        with col2:
            student_data['application_order'] = st.number_input("Application Order", min_value=0, max_value=9, value=1)

        with col3:
            student_data['course'] = st.number_input("Course", min_value=1, max_value=200, value=1)

        col1, col2, col3 = st.columns(3)

        with col1:
            student_data['daytime_evening_attendance'] = st.selectbox(
                "Attendance",
                options=[0, 1],
                format_func=lambda x: "Daytime" if x == 0 else "Evening",
                index=0
            )

        with col2:
            student_data['previous_qualification'] = st.number_input("Previous Qualification",
                                                                     min_value=1, max_value=50, value=1)

        with col3:
            student_data['previous_qualification_grade'] = st.number_input("Previous Qualification Grade",
                                                                           min_value=0.0, max_value=200.0, value=120.0)

        col1, col2, col3 = st.columns(3)

        with col1:
            student_data['nacionality'] = st.number_input("Nationality", min_value=1, max_value=200, value=1)

        with col2:
            student_data['admission_grade'] = st.number_input("Admission Grade",
                                                              min_value=0.0, max_value=200.0, value=120.0)

        with col3:
            student_data['displaced'] = st.selectbox(
                "Displaced",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                index=0
            )

        # Parents Information
        st.markdown("### 👨‍👩‍👧‍👦 Parents Information")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            student_data['mothers_qualification'] = st.number_input("Mother's Qualification",
                                                                    min_value=1, max_value=50, value=1)

        with col2:
            student_data['fathers_qualification'] = st.number_input("Father's Qualification",
                                                                    min_value=1, max_value=50, value=1)

        with col3:
            student_data['mothers_occupation'] = st.number_input("Mother's Occupation",
                                                                 min_value=0, max_value=200, value=1)

        with col4:
            student_data['fathers_occupation'] = st.number_input("Father's Occupation",
                                                                 min_value=0, max_value=200, value=1)

        # Financial Status
        st.markdown("### 💰 Financial Status")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            student_data['educational_special_needs'] = st.selectbox(
                "Educational Special Needs",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                index=0
            )

        with col2:
            student_data['debtor'] = st.selectbox(
                "Debtor",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                index=0
            )

        with col3:
            student_data['tuition_fees_up_to_date'] = st.selectbox(
                "Tuition Fees Up to Date",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                index=1
            )

        with col4:
            student_data['scholarship_holder'] = st.selectbox(
                "Scholarship Holder",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                index=0
            )

        col1, col2 = st.columns(2)

        with col1:
            student_data['international'] = st.selectbox(
                "International Student",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                index=0
            )

        # 1st Semester Performance
        st.markdown("### 📚 1st Semester Performance")
        col1, col2, col3 = st.columns(3)

        with col1:
            student_data['curricular_units_1st_sem_credited'] = st.number_input(
                "1st Sem Units Credited", min_value=0, max_value=30, value=0)
            student_data['curricular_units_1st_sem_enrolled'] = st.number_input(
                "1st Sem Units Enrolled", min_value=0, max_value=30, value=6)

        with col2:
            student_data['curricular_units_1st_sem_evaluations'] = st.number_input(
                "1st Sem Evaluations", min_value=0, max_value=30, value=6)
            student_data['curricular_units_1st_sem_approved'] = st.number_input(
                "1st Sem Units Approved", min_value=0, max_value=30, value=6)

        with col3:
            student_data['curricular_units_1st_sem_grade'] = st.number_input(
                "1st Sem Grade", min_value=0.0, max_value=20.0, value=10.0)
            student_data['curricular_units_1st_sem_without_evaluations'] = st.number_input(
                "1st Sem Without Evaluations", min_value=0, max_value=30, value=0)

        # 2nd Semester Performance
        st.markdown("### 📖 2nd Semester Performance")
        col1, col2, col3 = st.columns(3)

        with col1:
            student_data['curricular_units_2nd_sem_credited'] = st.number_input(
                "2nd Sem Units Credited", min_value=0, max_value=30, value=0)
            student_data['curricular_units_2nd_sem_enrolled'] = st.number_input(
                "2nd Sem Units Enrolled", min_value=0, max_value=30, value=6)

        with col2:
            student_data['curricular_units_2nd_sem_evaluations'] = st.number_input(
                "2nd Sem Evaluations", min_value=0, max_value=30, value=6)
            student_data['curricular_units_2nd_sem_approved'] = st.number_input(
                "2nd Sem Units Approved", min_value=0, max_value=30, value=6)

        with col3:
            student_data['curricular_units_2nd_sem_grade'] = st.number_input(
                "2nd Sem Grade", min_value=0.0, max_value=20.0, value=10.0)
            student_data['curricular_units_2nd_sem_without_evaluations'] = st.number_input(
                "2nd Sem Without Evaluations", min_value=0, max_value=30, value=0)

        # Economic Indicators
        st.markdown("### 📊 Economic Indicators")
        col1, col2, col3 = st.columns(3)

        with col1:
            student_data['unemployment_rate'] = st.number_input("Unemployment Rate (%)",
                                                                min_value=0.0, max_value=30.0, value=10.0)

        with col2:
            student_data['inflation_rate'] = st.number_input("Inflation Rate (%)",
                                                             min_value=-5.0, max_value=20.0, value=2.0)

        with col3:
            student_data['gdp'] = st.number_input("GDP Growth (%)",
                                                  min_value=-10.0, max_value=10.0, value=2.0)

        return student_data

    def calculate_engineered_features(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate additional engineered features

        Args:
            student_data: Dictionary berisi data student

        Returns:
            Dictionary dengan additional features
        """
        engineered_features = {}

        # First semester success rate
        if student_data['curricular_units_1st_sem_enrolled'] > 0:
            engineered_features['first_sem_success_rate'] = (
                    student_data['curricular_units_1st_sem_approved'] /
                    student_data['curricular_units_1st_sem_enrolled']
            )
        else:
            engineered_features['first_sem_success_rate'] = 0.0

        # Second semester success rate
        if student_data['curricular_units_2nd_sem_enrolled'] > 0:
            engineered_features['second_sem_success_rate'] = (
                    student_data['curricular_units_2nd_sem_approved'] /
                    student_data['curricular_units_2nd_sem_enrolled']
            )
        else:
            engineered_features['second_sem_success_rate'] = 0.0

        # Average grade
        grades = []
        if student_data['curricular_units_1st_sem_grade'] > 0:
            grades.append(student_data['curricular_units_1st_sem_grade'])
        if student_data['curricular_units_2nd_sem_grade'] > 0:
            grades.append(student_data['curricular_units_2nd_sem_grade'])

        engineered_features['average_grade'] = np.mean(grades) if grades else 0.0

        # Grade consistency (negative of absolute difference)
        if len(grades) == 2:
            engineered_features['grade_consistency'] = -abs(grades[0] - grades[1])
        else:
            engineered_features['grade_consistency'] = 0.0

        # Economic stress (combined economic indicator)
        engineered_features['economic_stress'] = (
                student_data['unemployment_rate'] +
                abs(student_data['inflation_rate']) -
                student_data['gdp']
        )

        # Financial support (inverse of being debtor + scholarship + tuition up to date)
        engineered_features['financial_support'] = (
                (1 - student_data['debtor']) +
                student_data['scholarship_holder'] +
                student_data['tuition_fees_up_to_date']
        )

        # Parent education features
        engineered_features['max_parent_education'] = max(
            student_data['mothers_qualification'],
            student_data['fathers_qualification']
        )

        engineered_features['min_parent_education'] = min(
            student_data['mothers_qualification'],
            student_data['fathers_qualification']
        )

        engineered_features['parent_education_gap'] = abs(
            student_data['mothers_qualification'] -
            student_data['fathers_qualification']
        )

        # Age category (0: young, 1: mature)
        engineered_features['age_category'] = 1 if student_data['age_at_enrollment'] >= 25 else 0

        # Grade improvement (2nd sem grade - 1st sem grade)
        engineered_features['grade_improvement'] = (
                student_data['curricular_units_2nd_sem_grade'] -
                student_data['curricular_units_1st_sem_grade']
        )

        return engineered_features

    def prepare_input_array(self, student_data: Dict[str, Any]) -> np.ndarray:
        """
        Convert student data dictionary ke numpy array untuk prediksi

        Args:
            student_data: Dictionary berisi data student

        Returns:
            numpy array dengan urutan features yang benar (47 features)
        """
        # Calculate engineered features
        engineered_features = self.calculate_engineered_features(student_data)

        # Urutan features sesuai dengan training model (47 features)
        input_array = np.array([[
            student_data['marital_status'],
            student_data['application_mode'],
            student_data['application_order'],
            student_data['course'],
            student_data['daytime_evening_attendance'],
            student_data['previous_qualification'],
            student_data['previous_qualification_grade'],
            student_data['nacionality'],
            student_data['mothers_qualification'],
            student_data['fathers_qualification'],
            student_data['mothers_occupation'],
            student_data['fathers_occupation'],
            student_data['admission_grade'],
            student_data['displaced'],
            student_data['educational_special_needs'],
            student_data['debtor'],
            student_data['tuition_fees_up_to_date'],
            student_data['gender'],
            student_data['scholarship_holder'],
            student_data['age_at_enrollment'],
            student_data['international'],
            student_data['curricular_units_1st_sem_credited'],
            student_data['curricular_units_1st_sem_enrolled'],
            student_data['curricular_units_1st_sem_evaluations'],
            student_data['curricular_units_1st_sem_approved'],
            student_data['curricular_units_1st_sem_grade'],
            student_data['curricular_units_1st_sem_without_evaluations'],
            student_data['curricular_units_2nd_sem_credited'],
            student_data['curricular_units_2nd_sem_enrolled'],
            student_data['curricular_units_2nd_sem_evaluations'],
            student_data['curricular_units_2nd_sem_approved'],
            student_data['curricular_units_2nd_sem_grade'],
            student_data['curricular_units_2nd_sem_without_evaluations'],
            student_data['unemployment_rate'],
            student_data['inflation_rate'],
            student_data['gdp'],
            # Additional engineered features (11 features)
            engineered_features['first_sem_success_rate'],
            engineered_features['second_sem_success_rate'],
            engineered_features['average_grade'],
            engineered_features['grade_consistency'],
            engineered_features['economic_stress'],
            engineered_features['financial_support'],
            engineered_features['max_parent_education'],
            engineered_features['min_parent_education'],
            engineered_features['parent_education_gap'],
            engineered_features['age_category'],
            engineered_features['grade_improvement']
        ]])

        return input_array

    def make_prediction(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Membuat prediksi dropout berdasarkan data student

        Args:
            student_data: Dictionary berisi data student

        Returns:
            Dictionary berisi hasil prediksi dan probabilitas
        """
        if not self.model_loaded:
            return {
                'success': False,
                'message': self.status_message,
                'prediction': None,
                'probability': None
            }

        try:
            # Prepare input array
            input_array = self.prepare_input_array(student_data)

            # Make prediction
            prediction = self.model.predict(input_array)[0]

            # Get prediction probability if available
            probability = None
            if hasattr(self.model, 'predict_proba'):
                prob_array = self.model.predict_proba(input_array)[0]
                probability = {
                    'dropout': float(prob_array[1]) if len(prob_array) > 1 else float(prob_array[0]),
                    'graduate': float(prob_array[0]) if len(prob_array) > 1 else 1 - float(prob_array[0])
                }

            # Interpret prediction
            prediction_text = "Dropout" if prediction == 1 else "Graduate"

            return {
                'success': True,
                'message': 'Prediction made successfully',
                'prediction': prediction,
                'prediction_text': prediction_text,
                'probability': probability
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Error making prediction: {str(e)}',
                'prediction': None,
                'probability': None
            }

    def display_prediction_results(self, results: Dict[str, Any]):
        """
        Display hasil prediksi dengan format yang menarik

        Args:
            results: Dictionary hasil prediksi dari make_prediction
        """
        if not results['success']:
            st.error(f"❌ {results['message']}")
            return

        st.success("✅ Prediction completed successfully!")

        # Display main prediction
        if results['prediction'] == 1:
            st.error(f"🚨 **Prediction: {results['prediction_text']}**")
            st.markdown("⚠️ This student is at **HIGH RISK** of dropping out.")
        else:
            st.success(f"🎓 **Prediction: {results['prediction_text']}**")
            st.markdown("✅ This student is likely to **GRADUATE** successfully.")

        # Display probabilities if available
        if results['probability']:
            st.markdown("### 📊 Prediction Probabilities")
            col1, col2 = st.columns(2)

            with col1:
                dropout_prob = results['probability']['dropout'] * 100
                st.metric(
                    label="Dropout Probability",
                    value=f"{dropout_prob:.1f}%",
                    delta=None
                )

            with col2:
                graduate_prob = results['probability']['graduate'] * 100
                st.metric(
                    label="Graduate Probability",
                    value=f"{graduate_prob:.1f}%",
                    delta=None
                )

            # Create probability chart
            fig, ax = plt.subplots(figsize=(8, 5))
            categories = ['Graduate', 'Dropout']
            probabilities = [results['probability']['graduate'] * 100,
                             results['probability']['dropout'] * 100]
            colors = ['#28a745', '#dc3545']

            bars = ax.bar(categories, probabilities, color=colors, alpha=0.7)
            ax.set_ylabel('Probability (%)')
            ax.set_title('Dropout vs Graduate Probability')
            ax.set_ylim(0, 100)

            # Add value labels on bars
            for bar, prob in zip(bars, probabilities):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')

            st.pyplot(fig)
            plt.close()

    def display_feature_analysis(self, student_data: Dict[str, Any]):
        """
        Display analysis of engineered features
        """
        st.markdown("### 🔍 Feature Analysis")

        engineered_features = self.calculate_engineered_features(student_data)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("1st Sem Success Rate", f"{engineered_features['first_sem_success_rate']:.2f}")
            st.metric("2nd Sem Success Rate", f"{engineered_features['second_sem_success_rate']:.2f}")
            st.metric("Average Grade", f"{engineered_features['average_grade']:.2f}")

        with col2:
            st.metric("Grade Consistency", f"{engineered_features['grade_consistency']:.2f}")
            st.metric("Economic Stress", f"{engineered_features['economic_stress']:.2f}")
            st.metric("Financial Support", f"{engineered_features['financial_support']:.0f}")

        with col3:
            st.metric("Max Parent Education", f"{engineered_features['max_parent_education']:.0f}")
            st.metric("Parent Education Gap", f"{engineered_features['parent_education_gap']:.0f}")
            age_cat = "Mature (25+)" if engineered_features['age_category'] == 1 else "Young (<25)"
            st.metric("Age Category", age_cat)


def main():
    """
    Main function untuk menjalankan Streamlit app
    """
    st.set_page_config(
        page_title="Student Dropout Predictor",
        page_icon="🎓",
        layout="wide"
    )

    st.title("🎓 Student Dropout Prediction System")
    st.markdown("---")

    # Initialize predictor
    predictor = StudentDropoutPredictor()

    # Show model status
    if predictor.model_loaded:
        st.success(f"✅ {predictor.status_message}")
        st.info("📊 Model now uses 47 features including engineered features for better prediction accuracy")
    else:
        st.error(f"❌ {predictor.status_message}")
        st.info("Please ensure the model file 'model.pkl' is in the same directory as this script.")
        st.stop()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Make Prediction", "🔍 Feature Analysis", "ℹ️ About"])

    with tab1:
        st.markdown("Fill in the student information below to predict dropout risk:")

        # Create input form
        student_data = predictor.create_input_form()

        # Prediction button
        if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
            with st.spinner("Making prediction..."):
                results = predictor.make_prediction(student_data)
                predictor.display_prediction_results(results)

    with tab2:
        st.markdown("View calculated engineered features:")

        # Create a sample form for feature analysis
        student_data = predictor.create_input_form()

        if st.button("📊 Analyze Features", type="secondary", use_container_width=True):
            predictor.display_feature_analysis(student_data)

    with tab3:
        st.markdown("""
        ## About This System

        This Student Dropout Prediction System uses machine learning to predict whether a student 
        is likely to drop out or graduate based on various factors including:

        ### Original Features (36)
        - **Personal Information**: Age, gender, marital status
        - **Academic Background**: Previous qualifications, grades, course information
        - **Family Background**: Parents' education and occupation
        - **Financial Status**: Scholarship, debt, tuition payment status
        - **Academic Performance**: 1st and 2nd semester performance metrics
        - **Economic Indicators**: Unemployment rate, inflation, GDP

        ### Engineered Features (11)
        - **First Semester Success Rate**: Ratio of approved to enrolled units
        - **Second Semester Success Rate**: Ratio of approved to enrolled units
        - **Average Grade**: Mean of semester grades
        - **Grade Consistency**: How consistent grades are between semesters
        - **Economic Stress**: Combined economic pressure indicator
        - **Financial Support**: Overall financial support level
        - **Max Parent Education**: Higher education level between parents
        - **Min Parent Education**: Lower education level between parents
        - **Parent Education Gap**: Difference in parent education levels
        - **Age Category**: Young vs mature student classification
        - **Grade Improvement**: Grade change from 1st to 2nd semester

        ### Model Features
        - Uses **47 total features** for prediction (36 original + 11 engineered)
        - Provides probability scores for both outcomes
        - Visual representation of prediction confidence
        - Feature analysis to understand key factors

        ### How to Use
        1. Fill in all the required information in the form
        2. Click "Make Prediction" to get results
        3. Use "Feature Analysis" tab to understand calculated features
        4. Review the prediction and probability scores

        ### Note
        This system is designed to help identify at-risk students early so that appropriate 
        interventions can be implemented to improve student retention rates.
        """)


if __name__ == "__main__":
    main()