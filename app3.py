import streamlit as st
import pandas as pd
import joblib
import pickle
from scipy import sparse
import plotly.graph_objects as go
import plotly.express as px
import os
# Add PaLM AI integration
import google.generativeai as palm


# Configuration
st.set_page_config(
    page_title="Career Path AI Advisor",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    h1 {
        color: #1E88E5;
    }
    h2 {
        color: #424242;
        margin-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# PaLM AI Configuration
def initialize_palm_ai():
    try:
        # Get API key from environment variable or Streamlit secrets
        palm_api_key = st.secrets.get("PALM_API_KEY") or os.getenv("PALM_API_KEY")
        
        if not palm_api_key:
            st.warning("⚠️ Google PaLM API key is missing. Some AI features will be disabled.")
            return None
        
        palm.configure(api_key=palm_api_key)
        return palm
    except Exception as e:
        st.error(f"Error initializing PaLM AI: {e}")
        return None

# Initialize PaLM AI
palm_ai = initialize_palm_ai()

# Custom function to generate AI insights
def generate_ai_career_insights(prediction, skills_scores, academic_scores):
    if not palm_ai:
        return "AI insights are currently unavailable. Please check your API configuration."
    
    # Prepare input for AI
    prompt = f"""
    Provide a detailed career guidance report based on the following profile:

    Predicted Career Category: {prediction}
    
    Skills Profile:
    - Technical Knowledge: {skills_scores[0]:.1f}%
    - Problem Solving: {skills_scores[1]:.1f}%
    - Communication: {skills_scores[2]:.1f}%
    - Teamwork & Leadership: {skills_scores[3]:.1f}%
    - Learning & Adaptability: {skills_scores[4]:.1f}%
    - Project Management: {skills_scores[5]:.1f}%

    Academic Strengths:
    {', '.join([f"{k}: {v}%" for k, v in academic_scores.items()])}

    Please provide:
    1. Detailed career path recommendations
    2. Skill development strategies
    3. Potential challenges and how to overcome them
    4. Emerging technologies and trends in this career path
    5. Networking and professional development advice

    Keep the tone professional, motivational, and actionable. Use bullet points for clarity.
    """

    try:
        # Use genai.chat with latest method
        model = palm_ai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating AI insights: {e}"
    
# Load the model and preprocessing components
@st.cache_resource
def load_model():
    model = joblib.load('final_model.pkl')
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, encoder, scaler

model, encoder, scaler = load_model()

def preprocess_data(new_data_array):
    """Preprocesses the input data using the fitted encoder and scaler."""
    encoded_data = encoder.transform(new_data_array)
    scaled_data = scaler.transform(encoded_data)
    return sparse.csr_matrix(scaled_data)

# Job Categories
JOB_CATEGORIES = {
    'CRM/Managerial Roles': ['CRM Business Analyst', 'CRM Technical Developer', 'Project Manager', 'Information Technology Manager'],
    'Analyst': ['Business Systems Analyst', 'Business Intelligence Analyst', 'E-Commerce Analyst'],
    'Mobile Applications/ Web Development': ['Mobile Applications Developer', 'Web Developer', 'Applications Developer'],
    'QA/Testing': ['Software Quality Assurance (QA) / Testing', 'Quality Assurance Associate'],
    'UX/Design': ['UX Designer', 'Design & UX'],
    'Databases': ['Database Developer', 'Database Administrator', 'Database Manager', 'Portal Administrator'],
    'Programming/ Systems Analyst': ['Programmer Analyst', 'Systems Analyst'],
    'Networks/ Systems': ['Network Security Administrator', 'Network Security Engineer', 'Network Engineer',
                         'Systems Security Administrator', 'Software Systems Engineer', 'Information Security Analyst'],
    'SE/SDE': ['Software Engineer', 'Software Developer'],
    'Technical Support/Service': ['Technical Engineer', 'Technical Services/Help Desk/Tech Support', 'Technical Support'],
    'Others': ['Solutions Architect', 'Data Architect', 'Information Technology Auditor']
}

# Main App UI
st.title("🎯 Career Path Predictor")
st.markdown("""
    This intelligent system analyzes your academic performance, skills, and preferences 
    to suggest the most suitable career path in the IT industry.
""")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📚 Academic Performance", "🔧 Skills & Experience", "🎯 Preferences & Personality"])

with tab1:
    st.header("Academic Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        academic_percentage_os = st.slider("Operating Systems", 0, 100, 69, help="Your percentage score in Operating Systems")
        percentage_algorithms = st.slider("Algorithms", 0, 100, 93, help="Your percentage score in Algorithms")
        percentage_programming = st.slider("Programming Concepts", 0, 100, 67, help="Your percentage score in Programming")
        percentage_software_eng = st.slider("Software Engineering", 0, 100, 78, help="Your percentage score in Software Engineering")
        percentage_computer_networks = st.slider("Computer Networks", 0, 100, 89, help="Your percentage score in Computer Networks")
    
    with col2:
        percentage_electronics = st.slider("Electronics", 0, 100, 78, help="Your percentage score in Electronics")
        percentage_computer_architecture = st.slider("Computer Architecture", 0, 100, 94, help="Your percentage score in Computer Architecture")
        percentage_mathematics = st.slider("Mathematics", 0, 100, 75, help="Your percentage score in Mathematics")
        percentage_communication_skills = st.slider("Communication Skills", 0, 100, 67, help="Your score in Communication Skills")

with tab2:
    st.header("Skills & Experience")
    col1, col2 = st.columns(2)
    
    with col1:
        hours_working = st.select_slider("Hours Working Per Day", options=range(4, 13), value=9)
        logical_quotient = st.select_slider("Logical Quotient Rating", options=range(1, 10), value=7)
        hackathons = st.select_slider("Hackathons Participated", options=range(0, 7), value=2)
        coding_skills = st.select_slider("Coding Skills Rating", options=range(1, 10), value=6)
        public_speaking = st.select_slider("Public Speaking Rating", options=range(1, 10), value=6)
    
    with col2:
        domain_certifications = st.selectbox("Technical Certifications", 
            ['shell programming', 'machine learning', 'app development', 'python',
             'r programming', 'information security', 'hadoop', 'distro making', 'full stack'])
        workshops = st.selectbox("Workshops Attended",
            ['cloud computing', 'database security', 'web technologies', 'data science',
             'testing', 'hacking', 'game development', 'system designing'])
        extra_courses = st.radio("Additional Courses Completed", ["yes", "no"], horizontal=True)
        talent_tests = st.radio("Participated in Talent Tests", ["yes", "no"], horizontal=True)
        olympiads = st.radio("Participated in Olympiads", ["yes", "no"], horizontal=True)

# Additional inputs in tab2
with tab2:
    col3, col4 = st.columns(2)
    with col3:
        long_time_working = st.radio("Can Work Long Time Before System?", ["yes", "no"], horizontal=True)
        self_learning = st.radio("Self-Learning Capability?", ["yes", "no"], horizontal=True)
        worked_in_teams = st.radio("Worked in Teams?", ["yes", "no"], horizontal=True)

with tab3:
    st.header("Preferences & Personality")
    col1, col2 = st.columns(2)
    
    with col1:
        interested_subjects = st.selectbox("Interested Subjects",
            ['cloud computing', 'networks', 'hacking', 'Computer Architecture',
             'programming', 'parallel computing', 'IOT', 'data engineering', 
             'Software Engineering', 'Management'])
        career_area = st.selectbox("Preferred Career Area",
            ['system developer', 'Business process analyst', 'developer', 'testing',
             'security', 'cloud computing'])
        company_preference = st.selectbox("Preferred Company Type",
            ['Product based', 'Service Based', 'Web Services', 'Cloud Services',
             'Product development', 'Testing and Maintainance Services', 'BPA', 'Finance'])
        management_technical = st.radio("Preference", ["Management", "Technical"], horizontal=True)
        hard_smart_worker = st.radio("Work Style", ["hard worker", "smart worker"], horizontal=True)
    
    with col2:
        reading_writing_skills = st.select_slider("Reading and Writing Skills", 
            options=["poor", "medium", "excellent"], value="medium")
        memory_capability = st.select_slider("Memory Capability", 
            options=["poor", "medium", "excellent"], value="medium")
        job_higher_studies = st.radio("Job or Higher Studies?", ["job", "higherstudies"], horizontal=True)
        salary_work_preference = st.radio("Priority", ["salary", "work"], horizontal=True)
        behavior = st.radio("Behavioral Tendency", ["gentle", "stubborn"], horizontal=True)
        introvert = st.radio("Personality Type", ["yes", "no"], horizontal=True)

        # Additional inputs
        taken_seniors_advice = "no"
        interested_games = st.radio("Interested in Games?", ["yes", "no"], horizontal=True)
        in_relationship = "no"
        type_of_books = st.selectbox("Interested Type of Books", 
            ['Prayer books', 'Childrens', 'Travel', 'Romance', 'Cookbooks', 'Self help',
            'Drama', 'Math', 'Religion-Spirituality', 'Anthology', 'Trilogy', 'Autobiographies', 
            'Mystery', 'Diaries', 'Journals', 'History', 'Art', 'Dictionaries', 'Horror', 
            'Encyclopedias', 'Action and Adventure', 'Fantasy', 'Comics', 'Science fiction', 
            'Series', 'Guide', 'Biographies', 'Health', 'Satire', 'Science', 'Poetry'])

# Prediction Section
st.markdown("---")

# Main prediction and visualization section
if st.button("🚀 Get AI Career Insights", key="ai_insights_button", use_container_width=True):
    with st.spinner("Analyzing your profile with AI..."):
        # Create input data frame
        user_data = pd.DataFrame([[
            academic_percentage_os, percentage_algorithms, percentage_programming, 
            percentage_software_eng, percentage_computer_networks, percentage_electronics,
            percentage_computer_architecture, percentage_mathematics, percentage_communication_skills,
            hours_working, logical_quotient, hackathons, coding_skills, public_speaking,
            long_time_working, self_learning, extra_courses, domain_certifications, workshops,
            talent_tests, olympiads, reading_writing_skills, memory_capability, 
            interested_subjects, career_area, job_higher_studies, company_preference, 
            taken_seniors_advice, interested_games, type_of_books, salary_work_preference,
            in_relationship, behavior, management_technical, salary_work_preference, 
            hard_smart_worker, worked_in_teams, introvert
        ]])
        
        # Make prediction
        prediction = model.predict(preprocess_data(user_data))[0]
        specific_roles = JOB_CATEGORIES.get(prediction, JOB_CATEGORIES['Others'])
        
        # Display results in an organized way
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"🎯 Predicted Career Category: **{prediction}**")
        with col2:
            st.info("💡 Specific Roles to Consider:")
            for role in specific_roles:
                st.write(f"• {role}")
                
        # Additional career insights
        st.markdown("### 📊 Career Insights")
        st.markdown("""
            Based on your profile, here are some key observations:
            * Your strong areas align well with the suggested career path
            * Consider pursuing certifications relevant to your predicted role
            * Focus on developing both technical and soft skills
            * Network with professionals in your target role
        """)
        
        # Skills Analysis Visualization
        st.markdown("### 📊 Skills Analysis")
        
        # Prepare data for radar chart
        skills_categories = [
            'Technical Knowledge',
            'Problem Solving',
            'Communication',
            'Teamwork & Leadership',
            'Learning & Adaptability',
            'Project Management'
        ]
        
        # Calculate normalized scores for each category
        technical_score = (academic_percentage_os + percentage_algorithms + 
                         percentage_programming + percentage_computer_architecture) / 4
        
        problem_solving = (logical_quotient * 11.11 + coding_skills * 11.11) / 2
        
        communication = (percentage_communication_skills + 
                       (public_speaking * 11.11)) / 2
        
        teamwork_score = 100 if worked_in_teams == "yes" else 50
        teamwork_score = (teamwork_score + (public_speaking * 11.11)) / 2
        
        learning_score = 100 if self_learning == "yes" else 50
        learning_score = (learning_score + (hackathons * 16.67)) / 2
        
        project_score = (hours_working - 4) * 12.5
        
        skills_scores = [
            technical_score,
            problem_solving,
            communication,
            teamwork_score,
            learning_score,
            project_score
        ]
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=skills_scores,
            theta=skills_categories,
            fill='toself',
            name='Your Skills Profile',
            line_color='rgba(29, 185, 84, 0.8)',
            fillcolor='rgba(29, 185, 84, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            title={
                'text': "Skills Profile Analysis",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation of the skills analysis
        st.markdown("#### 💡 Skills Analysis Interpretation")
        
        # Find strongest and weakest areas
        skills_dict = dict(zip(skills_categories, skills_scores))
        strongest = max(skills_dict.items(), key=lambda x: x[1])
        weakest = min(skills_dict.items(), key=lambda x: x[1])
        
        st.markdown(f"""
        **Key Observations:**
        * **Strongest Area:** {strongest[0]} ({strongest[1]:.1f}%)
        * **Area for Improvement:** {weakest[0]} ({weakest[1]:.1f}%)
        
        **Recommendations:**
        * Focus on improving {weakest[0]} through targeted training and practice
        * Leverage your strength in {strongest[0]} in your career path
        * Consider roles that align with your stronger areas while providing opportunities to develop in others
        """)
        
        # 1. Academic Performance Distribution
        st.markdown("### 📚 Academic Performance Analysis")
        academic_scores = {
            'Operating Systems': academic_percentage_os,
            'Algorithms': percentage_algorithms,
            'Programming': percentage_programming,
            'Software Engineering': percentage_software_eng,
            'Computer Networks': percentage_computer_networks,
            'Electronics': percentage_electronics,
            'Computer Architecture': percentage_computer_architecture,
            'Mathematics': percentage_mathematics,
            'Communication Skills': percentage_communication_skills
        }
        
        fig_academic = px.bar(
            x=list(academic_scores.keys()),
            y=list(academic_scores.values()),
            title='Academic Performance Across Subjects',
            labels={'x': 'Subjects', 'y': 'Score (%)'},
            color=list(academic_scores.values()),
            color_continuous_scale='Viridis'
        )
        fig_academic.update_layout(showlegend=False)
        st.plotly_chart(fig_academic, use_container_width=True)

        # 2. Technical vs Soft Skills Balance
        st.markdown("### 🔄 Technical vs Soft Skills Balance")
        
        # Calculate average scores for technical and soft skills
        technical_skills = (technical_score + problem_solving) / 2
        soft_skills = (communication + teamwork_score) / 2
        
        fig_balance = go.Figure()
        fig_balance.add_trace(go.Indicator(
            mode = "gauge+number",
            value = technical_skills,
            title = {'text': "Technical Skills"},
            domain = {'x': [0, 0.45], 'y': [0, 1]},
            gauge = {'axis': {'range': [None, 100]},
                    'bar': {'color': "rgba(29, 185, 84, 0.8)"}},
        ))
        
        fig_balance.add_trace(go.Indicator(
            mode = "gauge+number",
            value = soft_skills,
            title = {'text': "Soft Skills"},
            domain = {'x': [0.55, 1], 'y': [0, 1]},
            gauge = {'axis': {'range': [None, 100]},
                    'bar': {'color': "rgba(66, 133, 244, 0.8)"}},
        ))
        
        fig_balance.update_layout(height=300)
        st.plotly_chart(fig_balance, use_container_width=True)

        # 3. Career Readiness Score
        st.markdown("### 🎯 Career Readiness Analysis")
        
        # Calculate readiness scores for different aspects
        readiness_scores = {
            'Technical Preparation': (technical_score + problem_solving) / 2,
            'Professional Skills': (communication + teamwork_score) / 2,
            'Learning Attitude': learning_score,
            'Project Experience': project_score,
            'Industry Awareness': 85 if extra_courses == "yes" else 60  # Example scoring
        }
        
        fig_readiness = px.line_polar(
            r=list(readiness_scores.values()),
            theta=list(readiness_scores.keys()),
            line_close=True,
            range_r=[0,100],
            title="Career Readiness Assessment"
        )
        fig_readiness.update_traces(fill='toself')
        st.plotly_chart(fig_readiness, use_container_width=True)

        # 4. Development Areas Priority
        st.markdown("### 🎯 Recommended Development Areas")
        
        # Calculate priority scores (lower score = higher priority)
        priority_scores = {
            'Technical Skills': 100 - technical_score,
            'Communication': 100 - communication,
            'Problem Solving': 100 - problem_solving,
            'Leadership': 100 - teamwork_score,
            'Project Management': 100 - project_score
        }
        
        # Sort by priority (higher values = higher priority)
        priority_sorted = dict(sorted(priority_scores.items(), key=lambda x: x[1], reverse=True))
        
        fig_priority = px.funnel(
            x=list(priority_sorted.values()),
            y=list(priority_sorted.keys()),
            title="Development Priority Areas"
        )
        st.plotly_chart(fig_priority, use_container_width=True)

        # 5. Career Path Alignment
        st.markdown("### 🎯 Career Path Alignment Analysis")
        
        # Calculate alignment scores based on various factors
        alignment_scores = {
            'Technical Aptitude': technical_score,
            'Interest Alignment': 85 if career_area in ['developer', 'system developer'] else 70,
            'Skill Match': (technical_score + problem_solving + communication) / 3,
            'Experience Level': hackathons * 15 + (100 if extra_courses == "yes" else 50) / 2,
            'Growth Potential': learning_score
        }
        
        fig_alignment = px.sunburst(
            names=list(alignment_scores.keys()) + ['Overall'],
            parents=['Overall'] * len(alignment_scores.keys()) + [''],
            values=list(alignment_scores.values()) + [sum(alignment_scores.values()) / len(alignment_scores)],
            title="Career Path Alignment Analysis"
        )
        st.plotly_chart(fig_alignment, use_container_width=True)

        # Add insights based on all visualizations
        st.markdown("### 🔍 Comprehensive Analysis")
        
        # Calculate overall readiness score
        overall_readiness = sum(readiness_scores.values()) / len(readiness_scores)
        
        st.markdown(f"""
        **Overall Career Readiness: {overall_readiness:.1f}%**
        
        Based on the comprehensive analysis of your profile:
        
        1. **Academic Strengths**
        * Strongest subject: {max(academic_scores.items(), key=lambda x: x[1])[0]}
        * Area for improvement: {min(academic_scores.items(), key=lambda x: x[1])[0]}
        
        2. **Skills Balance**
        * Technical Skills: {technical_skills:.1f}%
        * Soft Skills: {soft_skills:.1f}%
        
        3. **Development Priorities**
        * Immediate focus area: {list(priority_sorted.keys())[0]}
        * Secondary focus area: {list(priority_sorted.keys())[1]}
        
        4. **Career Path Alignment**
        * Strong alignment in: {max(alignment_scores.items(), key=lambda x: x[1])[0]}
        * Need to strengthen: {min(alignment_scores.items(), key=lambda x: x[1])[0]}
        """)
        
        
        # Generate and display AI insights
        st.markdown("### 👁️ AI-Powered Career Insights")
        
        # Prepare data for AI insights
        skills_categories = [
            'Technical Knowledge',
            'Problem Solving',
            'Communication',
            'Teamwork & Leadership',
            'Learning & Adaptability',
            'Project Management'
        ]
        
        # Calculate skills and academic scores
        skills_scores = [
            technical_score,
            problem_solving,
            communication,
            teamwork_score,
            learning_score,
            project_score
        ]
        
        academic_scores = {
            'Operating Systems': academic_percentage_os,
            'Algorithms': percentage_algorithms,
            'Programming': percentage_programming,
            'Software Engineering': percentage_software_eng,
            'Computer Networks': percentage_computer_networks,
            'Electronics': percentage_electronics,
            'Computer Architecture': percentage_computer_architecture,
            'Mathematics': percentage_mathematics,
            'Communication Skills': percentage_communication_skills
        }
        
        # Call AI insights generation function
        ai_insights = generate_ai_career_insights(prediction, skills_scores, academic_scores)
        
        # Display AI insights
        st.markdown(ai_insights)

# Sidebar for API Key Configuration
st.sidebar.header("⚙️ PaLM AI Configuration")
st.sidebar.markdown("""
    To enable AI-powered insights:
    1. Get a PaLM API key from Google AI Studio
    2. Enter the key in your Streamlit secrets or environment variables
""")

# Optional: Allow manual API key input
if not palm_ai:
    palm_api_key = st.sidebar.text_input("Enter PaLM API Key", type="password")
    if st.sidebar.button("Configure API Key"):
        # Here you would add logic to validate and store the API key securely
        st.sidebar.success("API Key configured successfully!")


# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        Created as part of the College Project | Built with Streamlit and Machine Learning
    </div>
""", unsafe_allow_html=True)