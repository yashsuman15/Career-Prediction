import streamlit as st
import pandas as pd
import joblib
import pickle
from scipy import sparse

CRM_Managerial_Roles = ['CRM Business Analyst','CRM Technical Developer','Project Manager','Information Technology Manager']
Analyst = ['Business Systems Analyst','Business Intelligence Analyst','E-Commerce Analyst']
Mobile_Applications_Web_Development = ['Mobile Applications Developer','Web Developer','Applications Developer']
QA_Testing = ['Software Quality Assurance (QA) / Testing','Quality Assurance Associate']
UX_Design = ['UX Designer','Design & UX']
Databases = ['Database Developer','Database Administrator','Database Manager','Portal Administrator']
Programming_Systems_Analyst = ['Programmer Analyst','Systems Analyst']
Networks_Systems = ['Network Security Administrator','Network Security Engineer','Network Engineer',
                    'Systems Security Administrator','Software Systems Engineer','Information Security Analyst']
SE_SDE = ['Software Engineer','Software Developer']
Technical_Support_Service = ['Technical Engineer','Technical Services/Help Desk/Tech Support','Technical Support']
others = ['Solutions Architect','Data Architect','Information Technology Auditor']

# Load the trained model
model_path = 'final_model.pkl'  # Update with your actual model file
model = joblib.load(model_path)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def preprocess_data(new_data_array):
    """Preprocesses the new data array using the fitted encoder and scaler."""
    # One-Hot Encoding
    encoded_data = encoder.transform(new_data_array)

    # Scaling
    scaled_data = scaler.transform(encoded_data)

    # Conversion to Sparse Matrix
    preprocessed_data = sparse.csr_matrix(scaled_data)

    return preprocessed_data

# Title
st.title("Job Role Prediction System")

# User Inputs
st.header("Enter Your Details")

# Numeric inputs
academic_percentage_os = st.number_input("Academic Percentage in Operating Systems", min_value=0, max_value=100, value=69)
percentage_algorithms = st.number_input("Percentage in Algorithms", min_value=0, max_value=100, value=93)
percentage_programming = st.number_input("Percentage in Programming Concepts", min_value=0, max_value=100, value=67)
percentage_software_eng = st.number_input("Percentage in Software Engineering", min_value=0, max_value=100, value=78)
percentage_computer_networks = st.number_input("Percentage in Computer Networks", min_value=0, max_value=100, value=89)
percentage_electronics = st.number_input("Percentage in Electronics Subjects", min_value=0, max_value=100, value=78)
percentage_computer_architecture = st.number_input("Percentage in Computer Architecture", min_value=0, max_value=100, value=94)
percentage_mathematics = st.number_input("Percentage in Mathematics", min_value=0, max_value=100, value=75)
percentage_communication_skills = st.number_input("Percentage in Communication Skills", min_value=0, max_value=100, value=67)

# Categorical Inputs
hours_working = st.selectbox("Hours Working Per Day", [4, 5, 6, 7, 8, 9, 10, 11, 12], index=5)
logical_quotient = st.selectbox("Logical Quotient Rating", [1, 2, 3, 4, 5, 6, 7, 8, 9], index=6)
hackathons = st.selectbox("Hackathons Participated", [0, 1, 2, 3, 4, 5, 6], index=2)
coding_skills = st.selectbox("Coding Skills Rating", [1, 2, 3, 4, 5, 6, 7, 8, 9], index=5)
public_speaking = st.selectbox("Public Speaking Points", [1, 2, 3, 4, 5, 6, 7, 8, 9], index=5)

# Binary Inputs
long_time_working = st.radio("Can Work Long Time Before System?", ["yes", "no"], index=0)
self_learning = st.radio("Self-Learning Capability?", ["yes", "no"], index=0)
extra_courses = st.radio("Extra Courses Did?", ["yes", "no"], index=0)

domain_certifications = st.selectbox("Certifications", ['shell programming', 'machine learning', 'app development', 'python',
 'r programming', 'information security', 'hadoop', 'distro making', 'full stack'], index=1)
workshops = st.selectbox("Workshops", ['cloud computing', 'database security', 'web technologies', 'data science',
 'testing', 'hacking', 'game development', 'system designing'], index=3)

talent_tests = st.radio("Talent Tests Taken?", ["yes", "no"], index=1)
olympiads = st.radio("Olympiads Participated?", ["yes", "no"], index=1)

reading_writing_skills = st.selectbox("Reading and Writing Skills", ["excellent", "medium", "poor"], index=0)
memory_capability = st.selectbox("Memory Capability Score", ["excellent", "medium", "poor"], index=1)

interested_subjects = st.selectbox("Interested Subjects", ['cloud computing', 'networks', 'hacking', 'Computer Architecture',
 'programming', 'parallel computing', 'IOT', 'data engineering', 'Software Engineering', 'Management'], index=9)
career_area = st.selectbox("Interested Career Area", ['system developer', 'Business process analyst', 'developer', 'testing',
 'security', 'cloud computing'], index=1)
job_higher_studies = st.radio("Job or Higher Studies?", ['higherstudies', 'job'], index=0)
company_preference = st.selectbox("Type of Company to Settle In", ['Web Services', 'SAaS services', 'Sales and Marketing',
 'Testing and Maintainance Services', 'product development', 'BPA', 'Service Based', 'Product based', 'Cloud Services', 'Finance'], index=7)

taken_seniors_advice = st.radio("Taken Inputs from Seniors or Elders?", ['yes', 'no'], index=0)
interested_games = st.radio("Interested in Games?", ['yes', 'no'], index=1)
type_of_books = st.selectbox("Interested Type of Books", ['Prayer books', 'Childrens', 'Travel', 'Romance', 'Cookbooks', 'Self help',
 'Drama', 'Math', 'Religion-Spirituality', 'Anthology', 'Trilogy', 'Autobiographies', 'Mystery', 'Diaries', 'Journals', 'History', 'Art',
 'Dictionaries', 'Horror', 'Encyclopedias', 'Action and Adventure', 'Fantasy', 'Comics', 'Science fiction', 'Series', 'Guide',
 'Biographies', 'Health', 'Satire', 'Science', 'Poetry'], index=12)

salary_work_preference = st.radio("Salary or Work Preference?", ['salary', 'work'], index=0)
in_relationship = st.radio("In a Relationship?", ['yes', 'no'], index=0)
behavior = st.radio("Gentle or Tough Behavior?", ['stubborn', 'gentle'], index=1)
management_technical = st.radio("Management or Technical?", ['Management', 'Technical'], index=0)
hard_smart_worker = st.radio("Hard or Smart Worker?", ['hard worker', 'smart worker'], index=1)
worked_in_teams = st.radio("Worked in Teams?", ['yes', 'no'], index=0)
introvert = st.radio("Introvert?", ['yes', 'no'], index=0)

# Predict button
if st.button("Predict Job Role"):
    user_data = pd.DataFrame([[
        academic_percentage_os, 
        percentage_algorithms, 
        percentage_programming, 
        percentage_software_eng,
        percentage_computer_networks, 
        percentage_electronics, 
        percentage_computer_architecture,
        percentage_mathematics, 
        percentage_communication_skills, 
        hours_working, 
        logical_quotient, 
        hackathons,
        coding_skills, 
        public_speaking, 
        long_time_working, 
        self_learning, 
        extra_courses,
        domain_certifications, 
        workshops, 
        talent_tests, 
        olympiads, 
        reading_writing_skills, 
        memory_capability,
        interested_subjects, 
        career_area, 
        job_higher_studies, 
        company_preference, 
        taken_seniors_advice,
        interested_games, 
        type_of_books, 
        salary_work_preference, 
        in_relationship, 
        behavior, 
        management_technical,
        salary_work_preference, 
        hard_smart_worker, 
        worked_in_teams, 
        introvert
    ]])
    
    prediction = model.predict(preprocess_data(user_data))
    st.success(f"Suggested Job Role: {prediction}")
    
    if prediction[0] == 'CRM/Managerial Roles':
        cat=CRM_Managerial_Roles
    elif prediction[0] == 'Analyst':
        cat=Analyst
    elif prediction[0] == 'Mobile Applications/ Web Development':
        cat=Mobile_Applications_Web_Development
    elif prediction[0] == 'QA/Testing':
        cat=QA_Testing
    elif prediction[0] == 'UX/Design':
        cat=UX_Design
    elif prediction[0] == 'Databases':
        cat=Databases
    elif prediction[0] == 'Programming/ Systems Analyst':
        cat=Programming_Systems_Analyst
    elif prediction[0] == 'Networks/ Systems':
        cat=Networks_Systems
    elif prediction[0] == 'SE/SDE':
        cat=SE_SDE
    elif prediction[0] == 'Technical Support/Service':
        cat=Technical_Support_Service
    else:
        cat=others
        
    st.success(f"Suggested Job Role: {cat}")
    

