# ResumeInsight Pro - Enhanced Resume Analyzer

# FORCE NLTK DOWNLOAD BEFORE ANY IMPORTS - FIX FOR STREAMLIT CLOUD
import nltk
import os
import tempfile

# Use temp directory for NLTK data in Streamlit Cloud (writable location)
nltk_data_path = os.path.join(tempfile.gettempdir(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)

# Add to NLTK path and set environment variable
nltk.data.path.append(nltk_data_path)
os.environ['NLTK_DATA'] = nltk_data_path

# Download required NLTK data with error handling
try:
    # Check and download required NLTK datasets
    required_packages = ['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet']
    
    for package in required_packages:
        try:
            if package == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif package == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif package == 'averaged_perceptron_tagger':
                nltk.data.find('taggers/averaged_perceptron_tagger')
            elif package == 'wordnet':
                nltk.data.find('corpora/wordnet')
            print(f"✅ NLTK {package} already available")
        except LookupError:
            print(f"📥 Downloading NLTK {package}...")
            nltk.download(package, download_dir=nltk_data_path, quiet=True)
    
    print("✅ All NLTK data ready")
    
except Exception as e:
    print(f"⚠️ NLTK initialization warning: {e}")

# Now import other packages
import streamlit as st
import pandas as pd
import base64
import random
import time
import datetime
import io
import re
import json
from PyPDF2 import PdfReader
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter

# OCR features disabled for deployment
OCR_AVAILABLE = False

# FIX FOR SPACY MODEL ERROR
import spacy
import subprocess
import sys

# Set NLTK data path again before importing pyresparser
nltk.data.path.append(nltk_data_path)

try:
    # Try to load the spaCy model
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded successfully")
except OSError:
    print("📥 spaCy model not found, using fallback parsing")
    nlp = None
except Exception as e:
    print(f"⚠️ spaCy model error: {e}")
    nlp = None

# Safe import for pyresparser with error handling
try:
    from pyresparser import ResumeParser
    print("✅ pyresparser imported successfully")
except Exception as e:
    print(f"⚠️ pyresparser import issue: {e}")
    # Create a mock ResumeParser for fallback
    class MockResumeParser:
        def __init__(self, file_path):
            self.file_path = file_path
            
        def get_extracted_data(self):
            return {
                'name': '',
                'email': '',
                'mobile_number': '',
                'skills': [],
                'education': [],
                'experience': [],
                'company_names': [],
                'college_name': '',
                'designation': '',
                'total_experience': 0
            }
    
    ResumeParser = MockResumeParser
    print("⚠️ Using fallback resume parser")

# CUSTOM TAGS COMPONENT TO REPLACE STREAMLIT_TAGS
def custom_tags_input(label, value=None, suggestions=None, key=None):
    """
    Custom tags input component to replace streamlit_tags
    """
    if value is None:
        value = []
    
    if suggestions is None:
        suggestions = []
    
    # Create a unique key for session state
    tags_key = f"tags_{key}" if key else "tags_default"
    
    # Initialize session state
    if tags_key not in st.session_state:
        st.session_state[tags_key] = value
    
    # Display current tags
    st.write(f"**{label}**")
    
    # Show current tags as chips
    if st.session_state[tags_key]:
        tags_html = "<div style='margin-bottom: 10px;'>"
        for tag in st.session_state[tags_key]:
            tags_html += f"<span style='background-color: #667eea; color: white; padding: 4px 12px; margin: 2px; border-radius: 16px; display: inline-block; font-size: 14px;'>{tag}</span>"
        tags_html += "</div>"
        st.markdown(tags_html, unsafe_allow_html=True)
    
    # Input for new tag
    col1, col2 = st.columns([3, 1])
    with col1:
        new_tag = st.text_input("Add a skill (press Enter or click Add)", key=f"input_{key}")
    with col2:
        add_clicked = st.button("Add", key=f"add_{key}")
    
    # Add tag logic
    if (new_tag and new_tag.strip()) and (add_clicked or st.session_state.get(f"input_{key}_submitted", False)):
        tag_text = new_tag.strip()
        if tag_text and tag_text not in st.session_state[tags_key]:
            st.session_state[tags_key].append(tag_text)
        # Clear the input
        st.session_state[f"input_{key}"] = ""
    
    # Quick suggestions
    if suggestions and st.session_state[tags_key]:
        st.write("Quick add:")
        sugg_cols = st.columns(4)
        for idx, sugg in enumerate(suggestions[:4]):
            if sugg not in st.session_state[tags_key]:
                if sugg_cols[idx % 4].button(sugg, key=f"sugg_{key}_{idx}"):
                    st.session_state[tags_key].append(sugg)
                    st.rerun()
    
    # Remove tag option
    if st.session_state[tags_key]:
        st.write("Remove tags:")
        remove_cols = st.columns(4)
        for idx, tag in enumerate(st.session_state[tags_key]):
            if remove_cols[idx % 4].button(f"❌ {tag}", key=f"remove_{key}_{idx}"):
                st.session_state[tags_key].remove(tag)
                st.rerun()
    
    return st.session_state[tags_key]

from PIL import Image
import sqlite3  # Using SQLite for deployment

# Import courses - create a simple fallback if Courses.py doesn't exist
try:
    from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
except:
    # Fallback course data
    ds_course = [("Python for Data Science", "https://www.coursera.org/specializations/data-science-python"), 
                ("Machine Learning Basics", "https://www.coursera.org/learn/machine-learning"),
                ("Data Analysis with Python", "https://www.coursera.org/learn/data-analysis-with-python")]
    web_course = [("Web Development Fundamentals", "https://www.coursera.org/specializations/web-design"),
                 ("Full Stack Web Development", "https://www.coursera.org/specializations/full-stack-react"),
                 ("JavaScript Programming", "https://www.coursera.org/specializations/javascript-beginner")]
    android_course = [("Android Development", "https://www.coursera.org/learn/android-app-development"),
                     ("Kotlin for Android", "https://www.coursera.org/learn/kotlin-for-android"),
                     ("Mobile App Development", "https://www.coursera.org/specializations/android-app-development")]
    ios_course = [("iOS Development with Swift", "https://www.coursera.org/learn/ios-development-swift"),
                 ("Swift Programming", "https://www.coursera.org/learn/swift-programming"),
                 ("Mobile App Design", "https://www.coursera.org/learn/mobile-app-design")]
    uiux_course = [("UI/UX Design", "https://www.coursera.org/specializations/ui-ux-design"),
                  ("Interaction Design", "https://www.coursera.org/learn/interaction-design"),
                  ("User Research", "https://www.coursera.org/learn/user-research")]
    resume_videos = [("How to write a great resume", "https://youtube.com/watch?v=BYUyjn3fhV4"),
                    ("Resume tips for 2024", "https://youtube.com/watch?v=9hdz2i6e1yw"),
                    ("ATS Resume Guide", "https://youtube.com/watch?v=6tZfp8l7l_s")]
    interview_videos = [("Interview preparation tips", "https://youtube.com/watch?v=HG68Ymazo18"),
                       ("Technical Interview Guide", "https://youtube.com/watch?v=1qw5ITr3k9E"),
                       ("Behavioral Interview Questions", "https://youtube.com/watch?v=PJm6kdkKG94")]

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------------
# Enhanced Configuration
# -----------------------------
RESUME_TEMPLATES = {
    'professional': {
        'name': 'Professional',
        'description': 'Clean and corporate style',
        'color': '#2c3e50'
    },
    'modern': {
        'name': 'Modern', 
        'description': 'Contemporary design with accents',
        'color': '#667eea'
    },
    'creative': {
        'name': 'Creative',
        'description': 'For design and creative roles',
        'color': '#e74c3c'
    },
    'minimal': {
        'name': 'Minimal',
        'description': 'Simple and elegant',
        'color': '#27ae60'
    }
}

INTERVIEW_QUESTIONS = {
    'technical': [
        "Explain the concept of [skill] and how you've used it in projects",
        "What's your experience with [technology]?",
        "How do you approach debugging complex issues?",
        "Describe your experience with version control systems",
        "What programming paradigms are you most comfortable with?"
    ],
    'behavioral': [
        "Tell me about a time you faced a major challenge and how you overcame it",
        "Describe a situation where you had to work with a difficult team member",
        "How do you handle tight deadlines and pressure?",
        "Tell me about your greatest professional achievement",
        "Describe a time you made a mistake and what you learned from it"
    ],
    'general': [
        "Why do you want to work for our company?",
        "Where do you see yourself in 5 years?",
        "What are your strengths and weaknesses?",
        "Why should we hire you?",
        "What motivates you at work?"
    ]
}

SKILLS_IMPROVEMENT = {
    'Data Science': {
        'foundational': ['Python', 'Statistics', 'SQL', 'Data Visualization'],
        'intermediate': ['Machine Learning', 'Pandas', 'NumPy', 'Data Cleaning'],
        'advanced': ['Deep Learning', 'Big Data', 'MLOps', 'AWS/Azure'],
        'resources': ['Kaggle courses', 'Fast.ai', 'Coursera ML specialization']
    },
    'Web Development': {
        'foundational': ['HTML/CSS', 'JavaScript', 'Git', 'Responsive Design'],
        'intermediate': ['React/Angular/Vue', 'Node.js', 'REST APIs', 'Database Design'],
        'advanced': ['TypeScript', 'GraphQL', 'Microservices', 'Docker/Kubernetes'],
        'resources': ['FreeCodeCamp', 'The Odin Project', 'MDN Web Docs']
    },
    'Mobile Development': {
        'foundational': ['Java/Kotlin/Swift', 'UI/UX Principles', 'Mobile Design Patterns'],
        'intermediate': ['Flutter/React Native', 'REST APIs', 'State Management', 'Performance'],
        'advanced': ['Native Modules', 'CI/CD for Mobile', 'Security', 'App Store Deployment'],
        'resources': ['Android Developers', 'Apple Developer', 'Flutter Docs']
    },
    'UI/UX Design': {
        'foundational': ['Figma/Sketch', 'Design Principles', 'Wireframing', 'User Research'],
        'intermediate': ['Prototyping', 'Design Systems', 'Accessibility', 'Interaction Design'],
        'advanced': ['Design Thinking', 'UX Strategy', 'Metrics & Analytics', 'Leadership'],
        'resources': ['Interaction Design Foundation', 'Nielsen Norman Group', 'UX Collective']
    }
}

# -----------------------------
# Database setup - SIMPLIFIED FOR DEPLOYMENT
# -----------------------------
DB_NAME = 'CV'
TABLE_NAME = 'user_data'

def get_db_connection():
    """
    Use SQLite for deployment - no external database needed
    """
    try:
        conn = sqlite3.connect('resume_analyzer.db', check_same_thread=False)
        return conn
    except Exception as e:
        st.sidebar.error(f"❌ Database connection failed: {str(e)}")
        return None

# Initialize connection
connection = get_db_connection()

def init_database():
    """Initialize database tables"""
    if connection:
        try:
            cursor = connection.cursor()
            # Create table with SQLite compatible syntax
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Name VARCHAR(500),
                Email_ID VARCHAR(500),
                resume_score VARCHAR(8),
                Timestamp VARCHAR(50),
                Page_no VARCHAR(5),
                Predicted_Field TEXT,
                User_level TEXT,
                Actual_skills TEXT,
                Recommended_skills TEXT,
                Recommended_courses TEXT
            );""")
            connection.commit()
        except Exception as e:
            st.sidebar.warning(f"Database init warning: {str(e)}")

# Initialize database on startup
init_database()

def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills, courses):
    """Insert data into database"""
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute(f"""
            INSERT INTO {TABLE_NAME} 
            (Name, Email_ID, resume_score, Timestamp, Page_no, Predicted_Field, User_level, Actual_skills, Recommended_skills, Recommended_courses) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, email, str(res_score), timestamp, str(no_of_pages), reco_field, cand_level, skills, recommended_skills, courses))
            connection.commit()
            return True
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return False
    return False

# -----------------------------
# Helper Functions
# -----------------------------
def extract_email(text):
    m = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}", text)
    return m.group(0) if m else None

def extract_phone(text):
    m = re.search(r'(\+?\d[\d\s\-\(\)]{7,}\d)', text)
    return m.group(0).strip() if m else None

def extract_skills_from_text(text, known_skills=None):
    if known_skills is None:
        known_skills = {
            'python','java','c++','c#','javascript','react','angular','django','flask',
            'tensorflow','pytorch','keras','sql','mysql','postgresql','nosql','mongodb',
            'excel','git','docker','kubernetes','aws','azure','gcp','html','css','php',
            'android','kotlin','swift','flutter','figma','photoshop','linux','bash','rest api',
            'pandas','numpy','scikit-learn','opencv','nlp','tableau','power bi'
        }
    text_lower = text.lower()
    skills = set()
    header_pat = re.compile(r'(?P<header>(skills|technical skills|hard skills|skillset|skills & technologies))[:\n\r]*', re.IGNORECASE)
    for m in header_pat.finditer(text):
        start = m.end()
        block = text[start:start+800]
        bullets = re.findall(r'[\u2022\-\*\•]\s*([^\n\r]+)', block)
        if not bullets:
            candidates = re.split(r'[,;\n\r]', block)
            bullets = [c.strip() for c in candidates if len(c.strip())>1][:40]
        for b in bullets:
            for token in re.split(r'[\/,\|\-–]', b):
                token = token.strip()
                if not token:
                    continue
                token_low = token.lower()
                if token_low in known_skills:
                    skills.add(token.strip())
                else:
                    for ks in known_skills:
                        if ks in token_low:
                            skills.add(ks.capitalize())
        if skills:
            return sorted(skills)
    for ks in known_skills:
        if ks in text_lower:
            skills.add(ks.capitalize())
    return sorted(skills)

def show_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except:
        st.info("📄 PDF preview available in local environment only")

def extract_text_pypdf2(pdf_path_or_file):
    try:
        if hasattr(pdf_path_or_file, 'read'):
            # It's a file-like object
            reader = PdfReader(pdf_path_or_file)
        else:
            # It's a file path
            reader = PdfReader(pdf_path_or_file)
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text
    except:
        return ""

def pdfminer_extract_text(file_path):
    try:
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
        page_interpreter = PDFPageInterpreter(resource_manager, converter)
        with open(file_path, 'rb') as fh:
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
        converter.close()
        fake_file_handle.close()
        return text
    except:
        return ""

def course_recommender(course_list, key_suffix=""):
    st.subheader("**Courses & Certificates Recommendations 🎓**")
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5, key=f"course_slider_{key_suffix}")
    copy_courses = course_list[:]
    random.shuffle(copy_courses)
    for idx, (c_name, c_link) in enumerate(copy_courses, start=1):
        st.markdown(f"({idx}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if idx == no_of_reco:
            break
    return rec_course

def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'

# -----------------------------
# Enhanced Resume Scoring System
# -----------------------------
def enhanced_resume_scoring(resume_data, raw_text):
    """
    Advanced resume scoring with multiple factors
    """
    score = 0
    max_score = 100
    feedback = []
    
    # 1. Contact Information (15 points)
    contact_score = 0
    if resume_data.get('name'): contact_score += 5
    if resume_data.get('email'): contact_score += 5
    if resume_data.get('mobile_number'): contact_score += 5
    score += contact_score
    feedback.append(f"✅ Contact Info: {contact_score}/15 points")
    
    # 2. Skills Section (20 points)
    skills = resume_data.get('skills', [])
    skills_score = min(len(skills) * 2, 20)
    score += skills_score
    feedback.append(f"✅ Skills: {skills_score}/20 points ({len(skills)} skills found)")
    
    # 3. Experience (25 points)
    experience_score = 0
    experience = resume_data.get('experience', [])
    if experience:
        experience_score = min(len(experience) * 5, 25)
    score += experience_score
    feedback.append(f"✅ Experience: {experience_score}/25 points")
    
    # 4. Education (15 points)
    education_score = 0
    education = resume_data.get('education', [])
    if education:
        education_score = min(len(education) * 5, 15)
    score += education_score
    feedback.append(f"✅ Education: {education_score}/15 points")
    
    # 5. ATS Optimization (25 points)
    ats_score = 0
    
    # Check for important sections
    text_lower = raw_text.lower()
    important_sections = ['experience', 'education', 'skills', 'project', 'certification']
    found_sections = [section for section in important_sections if section in text_lower]
    ats_score += min(len(found_sections) * 3, 15)
    
    # Check for action verbs
    action_verbs = ['managed', 'developed', 'implemented', 'led', 'created', 'optimized']
    found_verbs = [verb for verb in action_verbs if verb in text_lower]
    ats_score += min(len(found_verbs) * 2, 10)
    
    score += ats_score
    feedback.append(f"✅ ATS Optimization: {ats_score}/25 points")
    
    # Determine grade
    if score >= 90:
        grade = "A+"
        color = "#27ae60"
    elif score >= 80:
        grade = "A"
        color = "#2ecc71"
    elif score >= 70:
        grade = "B+"
        color = "#f39c12"
    elif score >= 60:
        grade = "B"
        color = "#e67e22"
    else:
        grade = "C"
        color = "#e74c3c"
    
    return {
        'total_score': score,
        'grade': grade,
        'color': color,
        'feedback': feedback,
        'breakdown': {
            'contact': contact_score,
            'skills': skills_score,
            'experience': experience_score,
            'education': education_score,
            'ats': ats_score
        }
    }

# -----------------------------
# Video Functions
# -----------------------------
def get_youtube_thumbnail(video_id):
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

def extract_video_id(youtube_url):
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&?\n]+)',
        r'youtube\.com\/embed\/([^&?\n]+)',
        r'youtube\.com\/v\/([^&?\n]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    return None

def video_recommender_with_thumbnails(video_list, key_suffix=""):
    st.subheader("**🎥 Recommended Videos**")
    rec_videos = []
    no_of_reco = st.slider('Choose Number of Video Recommendations:', 1, 10, 4, key=f"video_slider_{key_suffix}")
    copy_videos = video_list[:]
    random.shuffle(copy_videos)
    
    for idx, video_item in enumerate(copy_videos, start=1):
        try:
            if isinstance(video_item, tuple) and len(video_item) == 2:
                v_name, v_link = video_item
            elif isinstance(video_item, str):
                v_link = video_item
                v_name = f"Video {idx}"
            elif isinstance(video_item, dict):
                v_name = video_item.get('name', video_item.get('title', 'Video'))
                v_link = video_item.get('link', video_item.get('url', '#'))
            else:
                v_name = f"Video {idx}"
                v_link = "#"
            
            video_id = extract_video_id(v_link)
            
            if video_id:
                col1, col2 = st.columns([1, 2])
                with col1:
                    thumbnail_url = get_youtube_thumbnail(video_id)
                    st.markdown(
                        f"""
                        <div style="position: relative; cursor: pointer;" onclick="window.open('{v_link}', '_blank')">
                            <img src="{thumbnail_url}" style="width: 100%; border-radius: 8px; border: 2px solid #667eea;">
                            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                                      background: rgba(0,0,0,0.7); border-radius: 50%; width: 50px; height: 50px; 
                                      display: flex; align-items: center; justify-content: center;">
                                <span style="color: white; font-size: 20px;">▶</span>
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(f"### {v_name}")
                    st.markdown(f"*Click to watch this video*")
                    st.markdown(f"[Open in YouTube]({v_link})")
            else:
                st.markdown(f"({idx}) [{v_name}]({v_link})")
            
            rec_videos.append(v_name)
            if idx == no_of_reco:
                break
            st.markdown("---")
        except Exception as e:
            st.warning(f"Could not process video: {str(e)}")
            continue
    return rec_videos

# -----------------------------
# NEW FEATURE: Cover Letter Generator
# -----------------------------
def cover_letter_generator():
    """Cover Letter Generator Interface"""
    st.header("📝 AI Cover Letter Generator")
    
    with st.form("cover_letter_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            applicant_name = st.text_input("Your Full Name*", placeholder="John Doe")
            applicant_email = st.text_input("Your Email*", placeholder="john.doe@email.com")
            applicant_phone = st.text_input("Your Phone", placeholder="+1 (555) 123-4567")
            job_title = st.text_input("Job Title you're applying for*", placeholder="Software Engineer")
            
        with col2:
            company = st.text_input("Company Name*", placeholder="Tech Innovations Inc.")
            hiring_manager = st.text_input("Hiring Manager Name (optional)", placeholder="Jane Smith")
            company_address = st.text_input("Company Address (optional)", placeholder="123 Business Ave, City, State")
            industry = st.selectbox("Industry", ["Technology", "Healthcare", "Finance", "Education", "Manufacturing", "Other"])
        
        st.subheader("Your Professional Background")
        key_skills = st.text_input("Key Skills*", placeholder="Python, Machine Learning, Data Analysis, Team Leadership")
        years_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=3)
        current_role = st.text_input("Current/Most Recent Role", placeholder="Senior Developer")
        
        st.subheader("Your Achievements")
        achievement_1 = st.text_area("Key Achievement 1*", placeholder="Led a team of 5 developers to deliver a project that increased company revenue by 20%...")
        achievement_2 = st.text_area("Key Achievement 2", placeholder="Optimized database queries reducing response time by 40%...")
        
        st.subheader("Company Specific Details")
        company_interest = st.text_area("Why you're interested in this company*", placeholder="I'm particularly drawn to Tech Innovations Inc. because of your innovative approach to...")
        company_field = st.text_input("Company's Field/Industry*", placeholder="Software development and AI solutions")
        matching_skills = st.text_input("Skills that match the job requirements*", placeholder="My experience in Python and team leadership aligns perfectly with your requirements for...")
        
        if st.form_submit_button("✨ Generate Cover Letter"):
            if applicant_name and job_title and company and key_skills and achievement_1 and company_interest:
                template_data = {
                    'date': datetime.datetime.now().strftime("%B %d, %Y"),
                    'applicant_name': applicant_name,
                    'applicant_contact': f"{applicant_email} | {applicant_phone}" if applicant_phone else applicant_email,
                    'hiring_manager': hiring_manager or "Hiring Manager",
                    'company': company,
                    'company_address': company_address or "",
                    'job_title': job_title,
                    'industry': industry,
                    'key_skills': key_skills,
                    'years_experience': years_experience,
                    'current_role': current_role,
                    'achievement_1': achievement_1,
                    'achievement_2': achievement_2,
                    'company_interest': company_interest,
                    'company_field': company_field,
                    'matching_skills': matching_skills,
                }
                
                cover_letter = generate_cover_letter_text(template_data)
                
                st.subheader("✨ Generated Cover Letter")
                st.text_area("Your Professional Cover Letter", cover_letter, height=400)
                
                # Download option
                b64 = base64.b64encode(cover_letter.encode()).decode()
                href = f'<a href="data:file/txt;base64,{b64}" download="cover_letter_{company}_{job_title}.txt">📥 Download Cover Letter</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                st.success("✅ Cover letter generated successfully! Remember to personalize it further before sending.")
            else:
                st.error("❌ Please fill in all required fields (marked with *)")

def generate_cover_letter_text(template_data):
    """Generate cover letter text from template data"""
    return f"""
{template_data['date']}

{template_data['hiring_manager']}
{template_data['company']}
{template_data['company_address']}

Dear {template_data['hiring_manager'].split()[0] if template_data['hiring_manager'] else 'Hiring Manager'},

I am writing to express my enthusiastic interest in the {template_data['job_title']} position at {template_data['company']}, as advertised on your company website. With {template_data['years_experience']} years of experience in {template_data['industry']} and expertise in {template_data['key_skills']}, I am confident that I possess the skills and experience necessary to excel in this role.

As a {template_data['current_role'] or 'professional'}, I have consistently demonstrated my ability to deliver exceptional results. For instance:
{template_data['achievement_1']}

{template_data['achievement_2'] if template_data['achievement_2'] else ''}

What particularly excites me about the opportunity at {template_data['company']} is {template_data['company_interest']}. I admire your company's work in {template_data['company_field']} and believe my skills in {template_data['matching_skills']} align perfectly with your requirements.

I am eager to bring my unique blend of skills and experience to {template_data['company']} and contribute to your ongoing success. I am confident that my background in {template_data['key_skills']} would make me a valuable asset to your team.

Thank you for considering my application. I have attached my resume for your review and would welcome the opportunity to discuss how my skills and experiences can contribute to the success of {template_data['company']}. I look forward to the possibility of speaking with you soon.

Sincerely,
{template_data['applicant_name']}
{template_data['applicant_contact']}
"""

# -----------------------------
# NEW FEATURE: Resume Templates
# -----------------------------
def resume_templates():
    """Resume Templates Gallery"""
    st.header("🎨 Professional Resume Templates")
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.2); margin: 10px 0; color: white;">
        <h3>Choose from our professionally designed resume templates</h3>
        <p>Each template is ATS-friendly and optimized for different industries and experience levels</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Template Gallery
    cols = st.columns(2)
    
    with cols[0]:
        # Professional Template
        st.subheader("💼 Professional")
        st.markdown("""
        **Ideal for:** Corporate roles, Business, Finance, Management
        **Features:**
        - Clean, traditional layout
        - ATS optimized
        - Professional typography
        - Balanced sections
        """)
        if st.button("Use Professional Template", key="professional_btn", use_container_width=True):
            st.session_state.selected_template = "professional"
            st.success("✅ Professional template selected! Navigate to Resume Builder to start building.")
        
        # Modern Template
        st.subheader("🚀 Modern")
        st.markdown("""
        **Ideal for:** Tech roles, Startups, Creative industries
        **Features:**
        - Contemporary design
        - Color accents
        - Skills visualization
        - Modern typography
        """)
        if st.button("Use Modern Template", key="modern_btn", use_container_width=True):
            st.session_state.selected_template = "modern"
            st.success("✅ Modern template selected! Navigate to Resume Builder to start building.")

    with cols[1]:
        # Creative Template
        st.subheader("🎨 Creative")
        st.markdown("""
        **Ideal for:** Designers, Artists, Marketing, UX/UI
        **Features:**
        - Visual appeal
        - Portfolio integration
        - Creative layouts
        - Color customization
        """)
        if st.button("Use Creative Template", key="creative_btn", use_container_width=True):
            st.session_state.selected_template = "creative"
            st.success("✅ Creative template selected! Navigate to Resume Builder to start building.")
        
        # Minimal Template
        st.subheader("📄 Minimal")
        st.markdown("""
        **Ideal for:** Academic roles, Research, Conservative industries
        **Features:**
        - Simple and clean
        - Maximum readability
        - Focus on content
        - One-page optimized
        """)
        if st.button("Use Minimal Template", key="minimal_btn", use_container_width=True):
            st.session_state.selected_template = "minimal"
            st.success("✅ Minimal template selected! Navigate to Resume Builder to start building.")
    
    # Template Preview Section
    st.markdown("---")
    st.subheader("📋 Template Features Comparison")
    
    feature_data = {
        'Feature': ['ATS Optimization', 'Modern Design', 'Color Options', 'Skills Chart', 'Portfolio Section', 'Mobile Friendly'],
        'Professional': ['✅', '✅', '🔵⚫', '✅', '❌', '✅'],
        'Modern': ['✅', '✅', '🔵🟣🟢', '✅', '✅', '✅'],
        'Creative': ['✅', '✅', 'Full Palette', '✅', '✅', '✅'],
        'Minimal': ['✅', '❌', '⚫⚪', '❌', '❌', '✅']
    }
    
    feature_df = pd.DataFrame(feature_data)
    st.dataframe(feature_df, use_container_width=True, hide_index=True)
    
    # Usage Tips
    st.markdown("---")
    st.subheader("💡 Template Selection Tips")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        **Choose Professional if:**
        - Applying to corporate companies
        - In finance/business roles
        - Prefer traditional format
        - Want maximum ATS compatibility
        """)
        
        st.markdown("""
        **Choose Modern if:**
        - In tech industry
        - Applying to startups
        - Want contemporary look
        - Have projects to showcase
        """)
    
    with tips_col2:
        st.markdown("""
        **Choose Creative if:**
        - Designer/artist role
        - Marketing position
        - Want visual impact
        - Have portfolio items
        """)
        
        st.markdown("""
        **Choose Minimal if:**
        - Academic applications
        - Research positions
        - Conservative industries
        - Need one-page resume
        """)

# -----------------------------
# NEW FEATURE: Interview Question Generator
# -----------------------------
def interview_question_generator():
    """Generate personalized interview questions"""
    st.header("🎤 AI Interview Question Generator")
    
    with st.form("interview_questions_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            job_role = st.text_input("Target Job Role*", placeholder="e.g., Data Scientist, Frontend Developer")
            experience_level = st.selectbox("Experience Level*", ["Entry Level (0-2 years)", "Mid Level (3-5 years)", "Senior Level (6+ years)", "Executive"])
            industry = st.selectbox("Industry*", ["Technology", "Finance", "Healthcare", "Education", "Manufacturing", "Startup", "Other"])
        
        with col2:
            question_types = st.multiselect("Question Types*", 
                                          ["Technical Skills", "Behavioral", "Situational", "Leadership", "Cultural Fit", "Problem Solving"],
                                          default=["Technical Skills", "Behavioral"])
            num_questions = st.slider("Number of Questions", 5, 25, 15)
            difficulty = st.select_slider("Difficulty Level", options=["Easy", "Medium", "Hard", "Mixed"])
        
        # Use text input for skills instead of custom_tags_input
        skills_input = st.text_input("Your Key Skills* (comma separated)", 
                                   placeholder="e.g., Python, JavaScript, React, AWS, SQL")
        
        specific_technologies = st.text_input("Specific Technologies/Tools", placeholder="e.g., React, TensorFlow, AWS, Docker")
        
        submitted = st.form_submit_button("🎯 Generate Interview Questions")
        
        if submitted:
            # Process skills from text input
            skills = [skill.strip() for skill in skills_input.split(',')] if skills_input else []
            
            # Validation
            missing_fields = []
            if not job_role:
                missing_fields.append("Job Role")
            if not skills_input:
                missing_fields.append("Key Skills")
            if not question_types:
                missing_fields.append("Question Types")
            
            if missing_fields:
                st.error(f"❌ Please fill in all required fields: {', '.join(missing_fields)}")
            else:
                questions = generate_personalized_questions(job_role, experience_level, skills, question_types, difficulty, specific_technologies, num_questions)
                display_questions(questions, job_role, experience_level)

def generate_personalized_questions(job_role, experience_level, skills, question_types, difficulty, specific_technologies, num_questions):
    """Generate personalized interview questions based on inputs"""
    questions = []
    
    # Technical questions based on skills
    if "Technical Skills" in question_types and skills:
        tech_questions = [
            f"Explain your experience with {skill} and provide an example of how you've used it in a project"
            for skill in skills[:4]  # Top 4 skills
        ]
        questions.extend([{'type': 'Technical', 'question': q, 'difficulty': 'Medium'} for q in tech_questions])
    
    # Technology-specific questions
    if specific_technologies:
        tech_list = [tech.strip() for tech in specific_technologies.split(',')]
        for tech in tech_list[:3]:
            questions.append({
                'type': 'Technical',
                'question': f"What challenges have you faced while working with {tech} and how did you overcome them?",
                'difficulty': 'Hard'
            })
    
    # Behavioral questions
    if "Behavioral" in question_types:
        behavioral_questions = [
            f"Tell me about a time you faced a significant challenge in your role as {job_role} and how you handled it",
            f"Describe a situation where you had to work with a difficult team member. What was the outcome?",
            "How do you handle tight deadlines and competing priorities?",
            "Tell me about a time you made a mistake at work. What did you learn from it?",
            "Describe your proudest professional achievement and why it matters to you"
        ]
        questions.extend([{'type': 'Behavioral', 'question': q, 'difficulty': 'Medium'} for q in behavioral_questions])
    
    # Leadership questions for senior roles
    if "Leadership" in question_types and any(level in experience_level for level in ['Senior', 'Executive']):
        leadership_questions = [
            "How do you approach mentoring junior team members?",
            "Describe your experience leading a team through a difficult project",
            "How do you handle conflict within your team?",
            "What's your strategy for delegating tasks effectively?"
        ]
        questions.extend([{'type': 'Leadership', 'question': q, 'difficulty': 'Hard'} for q in leadership_questions])
    
    # Problem-solving questions
    if "Problem Solving" in question_types:
        problem_questions = [
            "Describe your process for troubleshooting a complex technical issue",
            "How do you approach learning new technologies or skills?",
            "Tell me about a time you had to think creatively to solve a problem",
            "What's your strategy for breaking down large, complex problems?"
        ]
        questions.extend([{'type': 'Problem Solving', 'question': q, 'difficulty': 'Medium'} for q in problem_questions])
    
    # Cultural fit questions
    if "Cultural Fit" in question_types:
        cultural_questions = [
            "What type of work environment do you thrive in?",
            "How do you handle feedback and criticism?",
            "What motivates you to do your best work?",
            "Describe your ideal company culture"
        ]
        questions.extend([{'type': 'Cultural Fit', 'question': q, 'difficulty': 'Easy'} for q in cultural_questions])
    
    # Situational questions
    if "Situational" in question_types:
        situational_questions = [
            "What would you do if you disagreed with your manager's technical decision?",
            "How would you handle a situation where a project deadline was moved up unexpectedly?",
            "What would you do if you discovered a critical bug in production?",
            "How would you approach a situation where you had to learn a new technology quickly?"
        ]
        questions.extend([{'type': 'Situational', 'question': q, 'difficulty': 'Medium'} for q in situational_questions])
    
    # Fill remaining slots with role-specific questions
    role_specific_questions = {
        'Data Scientist': [
            "How do you validate your machine learning models?",
            "Explain the bias-variance tradeoff",
            "How do you handle missing data in your datasets?",
            "What metrics do you use to evaluate model performance?"
        ],
        'Web Developer': [
            "Explain your approach to responsive design",
            "How do you optimize website performance?",
            "Describe your experience with version control systems",
            "What's your process for testing your code?"
        ],
        'Software Engineer': [
            "Explain your software development methodology",
            "How do you approach code reviews?",
            "Describe your experience with testing methodologies",
            "What considerations do you make for security in your applications?"
        ],
        'Frontend Developer': [
            "How do you ensure cross-browser compatibility?",
            "What's your experience with modern JavaScript frameworks?",
            "How do you optimize frontend performance?",
            "Describe your approach to accessibility"
        ],
        'Backend Developer': [
            "How do you design scalable APIs?",
            "What's your experience with database optimization?",
            "How do you handle authentication and authorization?",
            "Describe your approach to API security"
        ],
        'Mobile Developer': [
            "What's your experience with native vs cross-platform development?",
            "How do you optimize mobile app performance?",
            "Describe your approach to mobile UI/UX",
            "How do you handle different screen sizes and orientations?"
        ]
    }
    
    for role, role_qs in role_specific_questions.items():
        if role.lower() in job_role.lower():
            questions.extend([{'type': 'Role-Specific', 'question': q, 'difficulty': 'Hard'} for q in role_qs])
            break
    
    # Apply difficulty filter
    if difficulty != "Mixed":
        questions = [q for q in questions if q['difficulty'] == difficulty]
    
    # Shuffle and limit to requested number
    random.shuffle(questions)
    return questions[:num_questions]

def display_questions(questions, job_role, experience_level):
    """Display generated questions in an organized way"""
    st.subheader(f"🎯 Personalized Interview Questions for {experience_level} {job_role}")
    st.success(f"Generated {len(questions)} questions tailored to your profile")
    
    # Categorize questions by type
    question_categories = {}
    for q in questions:
        if q['type'] not in question_categories:
            question_categories[q['type']] = []
        question_categories[q['type']].append(q)
    
    # Display by category
    for category, category_questions in question_categories.items():
        with st.expander(f"{category} Questions ({len(category_questions)})", expanded=True):
            for i, q in enumerate(category_questions, 1):
                st.markdown(f"**Q{i}:** {q['question']}")
                
                # Difficulty indicator
                difficulty_color = {
                    'Easy': '🟢',
                    'Medium': '🟡', 
                    'Hard': '🔴'
                }
                st.caption(f"Difficulty: {difficulty_color.get(q['difficulty'], '⚪')} {q['difficulty']}")
                
                # Answer practice area
                with st.expander("💬 Practice Your Answer", expanded=False):
                    st.text_area(
                        f"Type your answer here...", 
                        key=f"answer_{category}_{i}_{random.randint(1000,9999)}",
                        height=100,
                        placeholder="Structure your answer using the STAR method:\n- Situation: Set the context\n- Task: What needed to be done\n- Action: What you specifically did\n- Result: The outcome and what you learned"
                    )
                
                st.markdown("---")
    
    # Interview Tips Section
    st.markdown("---")
    st.subheader("💡 Expert Interview Preparation Tips")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        **🎯 Preparation Strategy:**
        - Research the company thoroughly
        - Practice answers aloud, not just in your head
        - Prepare 3-5 questions to ask the interviewer
        - Review the job description and match your skills
        """)
        
        st.markdown("""
        **💪 Technical Interviews:**
        - Practice coding on a whiteboard or paper
        - Explain your thought process clearly
        - Don't be afraid to ask clarifying questions
        - Test your solutions with examples
        """)
    
    with tips_col2:
        st.markdown("""
        **🌟 Behavioral Questions:**
        - Use the STAR method consistently
        - Focus on your specific contributions
        - Quantify results when possible
        - Be honest about challenges and learning
        """)
        
        st.markdown("""
        **🤝 Final Tips:**
        - Arrive 10-15 minutes early
        - Dress appropriately for the company culture
        - Maintain positive body language
        - Send thank you notes within 24 hours
        """)
    
    # Download questions as text
    questions_text = f"Interview Questions for {experience_level} {job_role}\n\n"
    for category, category_questions in question_categories.items():
        questions_text += f"{category}:\n"
        for i, q in enumerate(category_questions, 1):
            questions_text += f"{i}. {q['question']} ({q['difficulty']})\n"
        questions_text += "\n"
    
    b64 = base64.b64encode(questions_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="interview_questions_{job_role.replace(" ", "_")}.txt">📥 Download Questions as Text</a>'
    st.markdown(href, unsafe_allow_html=True)

# -----------------------------
# User Section
# -----------------------------
def run_user_section():
    """Enhanced User Section - FIXED FOR SPACY MODEL ERROR"""
    
    # Show friendly warning about limited features
    st.warning("""
    ⚠️ **Note:** Some advanced parsing features are currently limited. 
    Basic resume analysis is still available and functional.
    """)
    
    st.subheader("👤 Resume Analysis")
    user_name = st.text_input("Full Name", placeholder="Enter your full name")
    
    if not user_name:
        st.warning("⚠️ Please enter your name to proceed")
        return
    
    pdf_file = st.file_uploader("📄 Choose your Resume", type=["pdf"])
    
    if pdf_file and user_name.strip():
        # Create temporary directory for uploaded files
        upload_dir = './Uploaded_Resumes'
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, pdf_file.name)
        
        try:
            with open(save_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            
            # Show PDF (with error handling)
            try:
                show_pdf(save_path)
            except:
                st.info("📄 PDF preview available in local environment only")
            
            st.markdown("---")
            
            # Resume Analysis with spaCy error handling
            resume_data = {}
            raw_text = ""
            
            try:
                # First try advanced parsing
                resume_data = ResumeParser(save_path).get_extracted_data()
                st.success("✅ Advanced resume parsing completed")
            except Exception as e:
                st.warning(f"⚠️ Advanced parsing features limited: {str(e)}")
                # Fallback to basic parsing
                resume_data = {}
            
            # Always extract text for fallback analysis
            raw_text = extract_text_pypdf2(save_path)
            if not raw_text.strip():
                raw_text = pdfminer_extract_text(save_path)

            # Enhanced data extraction with fallbacks
            resume_data['email'] = resume_data.get('email') or extract_email(raw_text)
            resume_data['mobile_number'] = resume_data.get('mobile_number') or extract_phone(raw_text)
            resume_data['skills'] = resume_data.get('skills') or extract_skills_from_text(raw_text)
            resume_data['name'] = user_name

            try:
                p_reader = PdfReader(save_path)
                no_of_pages = len(p_reader.pages)
            except:
                no_of_pages = resume_data.get('no_of_pages') or 0

            # Enhanced Resume Scoring
            enhanced_score = enhanced_resume_scoring(resume_data, raw_text)
            
            # Display Results
            display_enhanced_results(resume_data, enhanced_score, raw_text, no_of_pages)
            
        except Exception as e:
            st.error(f"File processing error: {str(e)}")
            # Ultimate fallback
            try:
                st.info("🔄 Using basic text analysis...")
                raw_text = extract_text_pypdf2(pdf_file)
                if not raw_text.strip():
                    raw_text = ""
                
                # Basic resume data extraction
                resume_data = {
                    'email': extract_email(raw_text),
                    'mobile_number': extract_phone(raw_text),
                    'skills': extract_skills_from_text(raw_text),
                    'name': user_name,
                    'experience': [],
                    'education': []
                }
                
                no_of_pages = 1
                enhanced_score = enhanced_resume_scoring(resume_data, raw_text)
                display_enhanced_results(resume_data, enhanced_score, raw_text, no_of_pages)
                
            except Exception as parse_error:
                st.error(f"Error analyzing resume: {str(parse_error)}")
                st.info("💡 Try uploading a different PDF file or check the file format")

def display_enhanced_results(resume_data, enhanced_score, raw_text, no_of_pages):
    """Display enhanced analysis results"""
    # Personal Information Card
    st.markdown(
        f"""
        <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #667eea; margin: 10px 0; color: white;">
            <h3>👋 Hello {resume_data['name']}</h3>
            <p><strong>Name:</strong> {resume_data['name']}</p>
            <p><strong>Email:</strong> {resume_data.get('email', 'Not Found')}</p>
            <p><strong>Contact:</strong> {resume_data.get('mobile_number', 'Not Found')}</p>
            <p><strong>Resume Pages:</strong> {no_of_pages}</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Candidate level
    cand_level = ''
    level_color = ''
    if no_of_pages == 1:
        cand_level = "Fresher"
        level_color = "#ff6b6b"
    elif no_of_pages == 2:
        cand_level = "Intermediate"
        level_color = "#4ecdc4"
    elif no_of_pages >= 3:
        cand_level = "Experienced"
        level_color = "#45b7d1"
    
    st.markdown(
        f"""
        <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.2); margin: 10px 0; color: white;">
            <h3>🎯 Experience Level</h3>
            <h4 style='color: {level_color};'>You are at {cand_level} level!</h4>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Skills Section
    skills_list = resume_data.get('skills', [])
    st.markdown(
        """
        <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.2); margin: 10px 0; color: white;">
            <h3>💼 Your Current Skills</h3>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Display skills as chips using custom function
    if skills_list:
        skills_html = "<div style='margin-bottom: 20px;'>"
        for skill in skills_list:
            skills_html += f"<span style='background-color: #667eea; color: white; padding: 8px 16px; margin: 4px; border-radius: 20px; display: inline-block; font-size: 14px;'>{skill}</span>"
        skills_html += "</div>"
        st.markdown(skills_html, unsafe_allow_html=True)
    else:
        st.info("No skills detected. Try adding a 'Skills' section to your resume.")

    # Enhanced Resume Score
    st.markdown(
        f"""
        <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.2); margin: 10px 0; color: white;">
            <h3>📊 Enhanced Resume Analysis</h3>
            <h2 style='color: {enhanced_score["color"]}; text-align: center;'>Grade: {enhanced_score["grade"]} ({enhanced_score["total_score"]}/100)</h2>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Score Breakdown
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Contact Info", f"{enhanced_score['breakdown']['contact']}/15")
    with col2:
        st.metric("Skills", f"{enhanced_score['breakdown']['skills']}/20")
    with col3:
        st.metric("Experience", f"{enhanced_score['breakdown']['experience']}/25")
    with col4:
        st.metric("Education", f"{enhanced_score['breakdown']['education']}/15")
    with col5:
        st.metric("ATS Optimization", f"{enhanced_score['breakdown']['ats']}/25")

    # Progress bar for overall score
    my_bar = st.progress(0)
    for i in range(enhanced_score['total_score']):
        time.sleep(0.01)
        my_bar.progress(i + 1)

    # Course Recommendations
    ds_keyword = ['tensorflow','keras','pytorch','machine learning','flask','streamlit']
    web_keyword = ['react','django','node js','php','laravel','javascript','angular js']
    android_keyword = ['android','kotlin','flutter']
    ios_keyword = ['ios','swift','xcode']
    uiux_keyword = ['figma','photoshop','adobe xd','canva']

    reco_field = ''
    recommended_courses = []
    skills_lower = [s.lower() for s in skills_list]
    
    # Determine career field and show recommendations
    if any(k in skills_lower for k in ds_keyword):
        reco_field = "Data Science / ML"
        recommended_courses = course_recommender(ds_course, "ds")
    elif any(k in skills_lower for k in web_keyword):
        reco_field = "Web Development"
        recommended_courses = course_recommender(web_course, "web")
    elif any(k in skills_lower for k in android_keyword):
        reco_field = "Android Development"
        recommended_courses = course_recommender(android_course, "android")
    elif any(k in skills_lower for k in ios_keyword):
        reco_field = "iOS Development"
        recommended_courses = course_recommender(ios_course, "ios")
    elif any(k in skills_lower for k in uiux_keyword):
        reco_field = "UI/UX Designer"
        recommended_courses = course_recommender(uiux_course, "uiux")
    else:
        reco_field = "General/Other"
        recommended_courses = course_recommender(ds_course, "general")

    # Skills Improvement Section
    skills_improvement_recommendations(skills_list, reco_field)

    # VIDEO RECOMMENDATIONS
    try:
        # Try to use videos from Courses.py
        video_recommender_with_thumbnails(resume_videos, "resume")
    except Exception as e:
        st.warning("Using fallback resume videos")
        # Fallback video list
        RESUME_VIDEOS = [
            ("How to Write a WINNING Resume", "https://www.youtube.com/watch?v=XYZ123"),
            ("Resume Tips for Tech Jobs", "https://www.youtube.com/watch?v=XYZ124"),
            ("ATS Friendly Resume Guide", "https://www.youtube.com/watch?v=XYZ125")
        ]
        video_recommender_with_thumbnails(RESUME_VIDEOS, "resume_fallback")
    
    try:
        st.subheader("**🎤 Interview Preparation Videos**")
        # Try to use videos from Courses.py
        video_recommender_with_thumbnails(interview_videos, "interview")
    except Exception as e:
        st.warning("Using fallback interview videos")
        # Fallback video list
        INTERVIEW_VIDEOS = [
            ("Top 10 Interview Questions and Answers", "https://www.youtube.com/watch?v=XYZ128"),
            ("Technical Interview Preparation", "https://www.youtube.com/watch?v=XYZ129"),
            ("Behavioral Interview Tips", "https://www.youtube.com/watch?v=XYZ130")
        ]
        video_recommender_with_thumbnails(INTERVIEW_VIDEOS, "interview_fallback")

    recommended_skills = skills_list[:5] + ['Communication','Teamwork']

    # Insert data to database
    success = insert_data(resume_data['name'], resume_data.get('email', ''), enhanced_score['total_score'], 
                str(datetime.datetime.now()), no_of_pages, reco_field, cand_level, 
                ",".join(skills_list), ",".join(recommended_skills), ",".join(recommended_courses))
    
    if success:
        st.success("✅ Your resume analysis has been saved successfully!")
    else:
        st.info("📊 Analysis completed! (Data not saved due to database issue)")

# -----------------------------
# Skills Improvement Section
# -----------------------------
def skills_improvement_recommendations(resume_skills, career_field):
    """Provide detailed skills improvement recommendations"""
    st.header("🚀 Skills Improvement Plan")
    
    if career_field in SKILLS_IMPROVEMENT:
        field_skills = SKILLS_IMPROVEMENT[career_field]
        resume_skills_lower = [s.lower() for s in resume_skills]
        
        # Analyze current skill level
        foundational_missing = [s for s in field_skills['foundational'] if s.lower() not in resume_skills_lower]
        intermediate_missing = [s for s in field_skills['intermediate'] if s.lower() not in resume_skills_lower]
        advanced_missing = [s for s in field_skills['advanced'] if s.lower() not in resume_skills_lower]
        
        # Display skill development roadmap
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🌱 Foundational")
            st.markdown("**Build Your Base**")
            for skill in field_skills['foundational']:
                status = "✅" if skill.lower() in resume_skills_lower else "📚"
                st.write(f"{status} {skill}")
        
        with col2:
            st.subheader("🚀 Intermediate")
            st.markdown("**Expand Your Skills**")
            for skill in field_skills['intermediate']:
                status = "✅" if skill.lower() in resume_skills_lower else "📚"
                st.write(f"{status} {skill}")
        
        with col3:
            st.subheader("🏆 Advanced")
            st.markdown("**Master Your Craft**")
            for skill in field_skills['advanced']:
                status = "✅" if skill.lower() in resume_skills_lower else "📚"
                st.write(f"{status} {skill}")
        
        # Learning Resources
        st.markdown("---")
        st.subheader("📚 Recommended Learning Resources")
        
        for resource in field_skills['resources']:
            st.markdown(f"- **{resource}**")
        
        # Personalized learning path
        st.markdown("---")
        st.subheader("🎯 Your 30-Day Learning Plan")
        
        if foundational_missing:
            st.write("**Month 1: Master the Fundamentals**")
            for skill in foundational_missing[:2]:
                st.write(f"- Week 1-2: Complete {skill} fundamentals course")
                st.write(f"- Week 3-4: Build a small project using {skill}")
        
        if intermediate_missing:
            st.write("**Month 2: Intermediate Skills**")
            for skill in intermediate_missing[:2]:
                st.write(f"- Week 5-6: Advanced {skill} concepts")
                st.write(f"- Week 7-8: Real-world project implementation")
        
        if advanced_missing:
            st.write("**Month 3: Advanced Mastery**")
            for skill in advanced_missing[:1]:
                st.write(f"- Week 9-12: Master {skill} and contribute to open source")
    
    else:
        st.info("💡 Focus on these high-demand skills:")
        high_demand_skills = [
            "Python Programming", "Cloud Computing (AWS/Azure)", "Data Analysis", 
            "Machine Learning", "React/JavaScript", "SQL/Database Management",
            "DevOps Tools", "Agile Methodology", "Communication Skills"
        ]
        
        for skill in high_demand_skills:
            st.write(f"📚 {skill}")

# -----------------------------
# Admin Section
# -----------------------------
def run_admin_section():
    """Enhanced Admin Section with FIXED analytics"""
    st.subheader("👨‍💼 Admin Dashboard")
    st.success('Welcome to Admin Side')
    
    ad_user = st.text_input("Username", key="admin_user")
    ad_password = st.text_input("Password", type='password', key="admin_pass")
    
    if st.button('Login', key="admin_login"):
        if ad_user == 'arpitsharma' and ad_password == 'sharma123':
            st.success("Welcome Arpit!")
            display_enhanced_admin_analytics()
        else:
            st.error("Wrong ID & Password Provided")

def display_enhanced_admin_analytics():
    """Display enhanced admin analytics - UPDATED FOR DEPLOYMENT"""
    if not connection:
        st.error("❌ Database not available")
        return
        
    # Enhanced Metrics - UPDATED SQL queries
    st.markdown("### 📊 Advanced Analytics Dashboard")
    
    admin_col1, admin_col2, admin_col3, admin_col4 = st.columns(4)
    
    with admin_col1:
        try:
            total_users = pd.read_sql(f"SELECT COUNT(*) as count FROM {TABLE_NAME}", connection).iloc[0]['count']
            st.metric("Total Users", total_users)
        except:
            st.metric("Total Users", 0)
    
    with admin_col2:
        try:
            avg_score_result = pd.read_sql(f"SELECT AVG(CAST(resume_score as NUMERIC)) as avg_score FROM {TABLE_NAME}", connection)
            avg_score = avg_score_result.iloc[0]['avg_score']
            st.metric("Average Resume Score", f"{avg_score:.1f}" if avg_score is not None else "0")
        except:
            st.metric("Average Score", "N/A")
    
    with admin_col3:
        try:
            exp_levels = pd.read_sql(f"SELECT COUNT(*) as exp_count FROM {TABLE_NAME} WHERE User_level = 'Experienced'", connection).iloc[0]['exp_count']
            st.metric("Experienced Users", exp_levels)
        except:
            st.metric("Experienced Users", 0)
    
    with admin_col4:
        try:
            recent_users = pd.read_sql(f"SELECT COUNT(*) as count FROM {TABLE_NAME} WHERE date(Timestamp) >= date('now', '-7 days')", connection).iloc[0]['count']
            st.metric("New Users (7 days)", recent_users)
        except:
            st.metric("Recent Users", "N/A")

    # Enhanced Visualizations
    try:
        query = f'SELECT * FROM {TABLE_NAME};'
        plot_data = pd.read_sql(query, connection)
        
        if not plot_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Field Distribution")
                if 'Predicted_Field' in plot_data.columns:
                    field_counts = plot_data['Predicted_Field'].astype(str).value_counts()
                    if not field_counts.empty:
                        fig = px.pie(values=field_counts.values, names=field_counts.index, 
                                   title='Career Field Distribution')
                        st.plotly_chart(fig)
                    else:
                        st.info("No field data available for visualization")
            
            with col2:
                st.subheader("Experience Level")
                if 'User_level' in plot_data.columns:
                    level_counts = plot_data['User_level'].astype(str).value_counts()
                    if not level_counts.empty:
                        fig = px.bar(x=level_counts.index, y=level_counts.values,
                                   title='User Experience Levels', color=level_counts.index)
                        st.plotly_chart(fig)
                    else:
                        st.info("No experience level data available")

            # Score Distribution
            st.subheader("Resume Score Distribution")
            if 'resume_score' in plot_data.columns:
                try:
                    scores = pd.to_numeric(plot_data['resume_score'], errors='coerce').dropna()
                    if not scores.empty:
                        fig = px.histogram(scores, nbins=20, title='Resume Score Distribution',
                                         labels={'value': 'Score', 'count': 'Number of Users'})
                        st.plotly_chart(fig)
                    else:
                        st.info("No score data available for visualization")
                except Exception as e:
                    st.info("Could not generate score distribution chart")
    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")

    # Data Export
    try:
        cursor = connection.cursor()
        cursor.execute(f'SELECT * FROM {TABLE_NAME}')
        data = cursor.fetchall()
        
        # Get column names
        if connection:
            try:
                # For SQLite
                cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
                columns = [column[1] for column in cursor.fetchall()]
            except:
                # For PostgreSQL - simplified approach
                columns = ['ID', 'Name', 'Email_ID', 'resume_score', 'Timestamp', 'Page_no', 
                          'Predicted_Field', 'User_level', 'Actual_skills', 'Recommended_skills', 'Recommended_courses']
        
        df = pd.DataFrame(data, columns=columns)
        
        st.subheader("User Data")
        st.dataframe(df)
        
        st.markdown(get_table_download_link(df, 'User_Data.csv', '📥 Download CSV Report'), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error exporting data: {str(e)}")

# -----------------------------
# Resume Builder Functions - FIXED
# -----------------------------
def init_resume_builder_session():
    """Initialize resume builder session state"""
    if 'resume_data' not in st.session_state:
        st.session_state.resume_data = {
            'personal_info': {'name': '', 'email': '', 'phone': '', 'linkedin': '', 'github': ''},
            'summary': '',
            'experience': [],
            'education': [],
            'skills': [],
            'projects': [],
            'certifications': []
        }

def render_resume_builder():
    """Render the resume builder interface"""
    st.header("📝 AI Resume Builder")
    
    init_resume_builder_session()
    
    # Check if template is selected
    if 'selected_template' in st.session_state:
        st.success(f"🎨 Using {st.session_state.selected_template} template")
    
    # Personal Information
    with st.expander("👤 Personal Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.resume_data['personal_info']['name'] = st.text_input(
                "Full Name*", 
                value=st.session_state.resume_data['personal_info']['name'],
                placeholder="John Doe"
            )
            st.session_state.resume_data['personal_info']['email'] = st.text_input(
                "Email*", 
                value=st.session_state.resume_data['personal_info']['email'],
                placeholder="john.doe@email.com"
            )
        with col2:
            st.session_state.resume_data['personal_info']['phone'] = st.text_input(
                "Phone", 
                value=st.session_state.resume_data['personal_info']['phone'],
                placeholder="+1 (555) 123-4567"
            )
            st.session_state.resume_data['personal_info']['linkedin'] = st.text_input(
                "LinkedIn", 
                value=st.session_state.resume_data['personal_info']['linkedin'],
                placeholder="linkedin.com/in/johndoe"
            )
    
    # Professional Summary
    with st.expander("📄 Professional Summary"):
        st.session_state.resume_data['summary'] = st.text_area(
            "Summary*",
            value=st.session_state.resume_data['summary'],
            placeholder="Experienced software developer with 5+ years in web development...",
            height=100
        )
    
    # Experience - FIXED DATE FORMAT AND ADD FUNCTIONALITY
    with st.expander("💼 Work Experience"):
        st.subheader("Add Work Experience")
        
        # Use a form key to ensure form resets properly
        with st.form("experience_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                company = st.text_input("Company*", key="exp_company")
                position = st.text_input("Position*", key="exp_position")
            with col2:
                # Fixed date format - use text input with better placeholder
                start_date = st.text_input("Start Date* (MM/YYYY)", placeholder="01/2020", key="exp_start")
                end_date = st.text_input("End Date (MM/YYYY) or 'Present'", placeholder="12/2023 or Present", key="exp_end")
            
            description = st.text_area("Description*", height=100, placeholder="Describe your responsibilities and achievements...", key="exp_desc")
            
            submitted = st.form_submit_button("Add Experience")
            
            if submitted:
                # Validation
                errors = []
                
                if not company:
                    errors.append("Company name is required")
                if not position:
                    errors.append("Position is required")
                if not start_date:
                    errors.append("Start date is required")
                if not description:
                    errors.append("Description is required")
                
                # Date validation - FIXED
                if start_date and start_date.lower() != 'present':
                    if not validate_date_format(start_date):
                        errors.append("Start date must be in MM/YYYY format (e.g., 01/2020)")
                
                if end_date and end_date.lower() != 'present':
                    if not validate_date_format(end_date) and end_date.lower() != 'present':
                        errors.append("End date must be in MM/YYYY format (e.g., 12/2023) or 'Present'")
                
                # Date logic validation
                if (start_date and end_date and 
                    start_date.lower() != 'present' and end_date.lower() != 'present' and
                    validate_date_format(start_date) and validate_date_format(end_date)):
                    
                    start_month, start_year = map(int, start_date.split('/'))
                    end_month, end_year = map(int, end_date.split('/'))
                    
                    if (end_year < start_year) or (end_year == start_year and end_month < start_month):
                        errors.append("End date cannot be before start date")
                
                if errors:
                    for error in errors:
                        st.error(f"❌ {error}")
                else:
                    # Add experience to session state
                    st.session_state.resume_data['experience'].append({
                        'company': company,
                        'position': position,
                        'start_date': start_date,
                        'end_date': end_date,
                        'description': description
                    })
                    st.success("✅ Experience added successfully!")
                    st.rerun()  # Force rerun to update the display
        
        # Display experiences
        if st.session_state.resume_data['experience']:
            st.subheader("Your Work Experience")
            for i, exp in enumerate(st.session_state.resume_data['experience']):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{exp['position']}** at **{exp['company']}**")
                    st.markdown(f"*{exp['start_date']} - {exp['end_date']}*")
                    st.write(exp['description'])
                with col2:
                    if st.button(f"🗑️ Remove", key=f"remove_exp_{i}"):
                        st.session_state.resume_data['experience'].pop(i)
                        st.rerun()
                st.markdown("---")
        else:
            st.info("💡 Add your work experience to build a comprehensive resume")
    
    # Education Section - FIXED DATE FORMAT
    with st.expander("🎓 Education"):
        st.subheader("Add Education")
        
        with st.form("education_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                institution = st.text_input("Institution/University*", key="edu_institution")
                degree = st.text_input("Degree/Certificate*", key="edu_degree")
            with col2:
                ed_start_date = st.text_input("Start Date* (MM/YYYY)", placeholder="08/2018", key="edu_start")
                ed_end_date = st.text_input("End Date (MM/YYYY) or 'Present'", placeholder="05/2022", key="edu_end")
            
            ed_description = st.text_area("Description/Achievements", height=100, key="edu_desc", 
                                        placeholder="GPA, honors, relevant coursework, achievements...")
            
            submitted = st.form_submit_button("Add Education")
            
            if submitted:
                # Validation
                errors = []
                
                if not institution:
                    errors.append("Institution name is required")
                if not degree:
                    errors.append("Degree is required")
                if not ed_start_date:
                    errors.append("Start date is required")
                
                # Date validation - FIXED
                if ed_start_date and ed_start_date.lower() != 'present':
                    if not validate_date_format(ed_start_date):
                        errors.append("Start date must be in MM/YYYY format (e.g., 08/2018)")
                
                if ed_end_date and ed_end_date.lower() != 'present':
                    if not validate_date_format(ed_end_date) and ed_end_date.lower() != 'present':
                        errors.append("End date must be in MM/YYYY format (e.g., 05/2022) or 'Present'")
                
                # Date logic validation
                if (ed_start_date and ed_end_date and 
                    ed_start_date.lower() != 'present' and ed_end_date.lower() != 'present' and
                    validate_date_format(ed_start_date) and validate_date_format(ed_end_date)):
                    
                    start_month, start_year = map(int, ed_start_date.split('/'))
                    end_month, end_year = map(int, ed_end_date.split('/'))
                    
                    if (end_year < start_year) or (end_year == start_year and end_month < start_month):
                        errors.append("End date cannot be before start date")
                
                if errors:
                    for error in errors:
                        st.error(f"❌ {error}")
                else:
                    st.session_state.resume_data['education'].append({
                        'institution': institution,
                        'degree': degree,
                        'start_date': ed_start_date,
                        'end_date': ed_end_date,
                        'description': ed_description
                    })
                    st.success("✅ Education added successfully!")
                    st.rerun()
        
        # Display education entries
        if st.session_state.resume_data['education']:
            st.subheader("Your Education")
            for i, edu in enumerate(st.session_state.resume_data['education']):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{edu['degree']}**")
                    st.markdown(f"*{edu['institution']}*")
                    st.markdown(f"*{edu['start_date']} - {edu['end_date']}*")
                    if edu['description']:
                        st.write(edu['description'])
                with col2:
                    if st.button(f"🗑️ Remove", key=f"remove_edu_{i}"):
                        st.session_state.resume_data['education'].pop(i)
                        st.rerun()
                st.markdown("---")
        else:
            st.info("💡 Add your education history to build a comprehensive resume")
    
    # Skills
    with st.expander("🛠️ Skills"):
        st.subheader("Add Your Skills")
        
        # Skills input with categories
        skill_categories = {
            "Technical Skills": ["Python", "JavaScript", "Java", "C++", "SQL", "React", "Node.js", "AWS", "Docker", "Git"],
            "Soft Skills": ["Communication", "Leadership", "Teamwork", "Problem Solving", "Time Management", "Adaptability"],
            "Tools & Technologies": ["VS Code", "Figma", "Jira", "Slack", "Photoshop", "Excel"]
        }
        
        selected_category = st.selectbox("Skill Category", list(skill_categories.keys()))
        
        col1, col2 = st.columns([3, 1])
        with col1:
            new_skill = st.text_input("Add Skill", placeholder="Enter a skill or select from suggestions", key="new_skill_input")
        with col2:
            add_skill_btn = st.button("Add Skill", use_container_width=True, key="add_skill_btn")
        
        # Quick suggestions
        st.write("Quick add:")
        sugg_cols = st.columns(4)
        for idx, skill in enumerate(skill_categories[selected_category][:4]):
            if sugg_cols[idx].button(skill, key=f"sugg_{selected_category}_{idx}"):
                if skill not in st.session_state.resume_data['skills']:
                    st.session_state.resume_data['skills'].append(skill)
                    st.rerun()
        
        if add_skill_btn and new_skill:
            skill_text = new_skill.strip()
            if skill_text and skill_text not in st.session_state.resume_data['skills']:
                st.session_state.resume_data['skills'].append(skill_text)
                st.rerun()
        
        # Display current skills
        if st.session_state.resume_data['skills']:
            st.subheader("Your Skills")
            skills_html = "<div style='margin-bottom: 20px;'>"
            for skill in st.session_state.resume_data['skills']:
                skills_html += f"<span style='background-color: #667eea; color: white; padding: 8px 16px; margin: 4px; border-radius: 20px; display: inline-block; font-size: 14px;'>{skill}</span>"
            skills_html += "</div>"
            st.markdown(skills_html, unsafe_allow_html=True)
            
            # Remove skills
            st.write("Remove skills:")
            remove_cols = st.columns(4)
            for idx, skill in enumerate(st.session_state.resume_data['skills']):
                if remove_cols[idx % 4].button(f"❌ {skill}", key=f"remove_skill_{idx}"):
                    st.session_state.resume_data['skills'].remove(skill)
                    st.rerun()
        else:
            st.info("💡 Add your skills to showcase your capabilities")
    
    # Projects Section
    with st.expander("🚀 Projects"):
        st.subheader("Add Projects")
        with st.form("project_form", clear_on_submit=True):
            project_name = st.text_input("Project Name*", key="proj_name")
            project_technologies = st.text_input("Technologies Used*", placeholder="Python, React, MongoDB, etc.", key="proj_tech")
            project_description = st.text_area("Project Description*", height=100, 
                                             placeholder="Describe the project, your role, and key achievements...", key="proj_desc")
            project_link = st.text_input("Project Link (optional)", placeholder="GitHub link, live demo, etc.", key="proj_link")
            
            submitted = st.form_submit_button("Add Project")
            
            if submitted:
                if project_name and project_technologies and project_description:
                    st.session_state.resume_data['projects'].append({
                        'name': project_name,
                        'technologies': project_technologies,
                        'description': project_description,
                        'link': project_link
                    })
                    st.success("✅ Project added successfully!")
                    st.rerun()
                else:
                    st.error("❌ Please fill in all required fields")
        
        # Display projects
        if st.session_state.resume_data['projects']:
            st.subheader("Your Projects")
            for i, project in enumerate(st.session_state.resume_data['projects']):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{project['name']}**")
                    st.markdown(f"*Technologies: {project['technologies']}*")
                    st.write(project['description'])
                    if project['link']:
                        st.markdown(f"[View Project]({project['link']})")
                with col2:
                    if st.button(f"🗑️ Remove", key=f"remove_project_{i}"):
                        st.session_state.resume_data['projects'].pop(i)
                        st.rerun()
                st.markdown("---")
    
    # Resume Preview and Export
    st.markdown("---")
    st.subheader("👀 Resume Preview")
    
    # Basic validation for required fields
    required_fields_filled = (
        st.session_state.resume_data['personal_info']['name'] and
        st.session_state.resume_data['personal_info']['email'] and
        st.session_state.resume_data['summary']
    )
    
    if not required_fields_filled:
        st.warning("⚠️ Please fill in all required fields (marked with *) to enable export")
    
    # Preview current resume data
    with st.expander("📋 Preview Resume Data"):
        st.json(st.session_state.resume_data)
    
    # Export Options - FIXED PDF GENERATION
    st.subheader("📤 Export Your Resume")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Generate PDF", use_container_width=True, disabled=not required_fields_filled):
            if required_fields_filled:
                # Generate and download PDF
                pdf_content = generate_pdf_resume(st.session_state.resume_data)
                b64 = base64.b64encode(pdf_content).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="resume_{st.session_state.resume_data["personal_info"]["name"].replace(" ", "_")}.pdf">📥 Download PDF Resume</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("✅ PDF resume generated successfully!")
            else:
                st.warning("Please fill in all required fields first")
    
    with col2:
        if st.button("📊 Download as JSON", use_container_width=True, disabled=not required_fields_filled):
            if required_fields_filled:
                # Create download link for JSON
                json_str = json.dumps(st.session_state.resume_data, indent=2, ensure_ascii=False)
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:file/json;base64,{b64}" download="resume_data.json">📥 Download JSON Resume</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("✅ Resume data exported as JSON!")
            else:
                st.warning("Please fill in all required fields first")
    
    with col3:
        if st.button("🔄 Reset Form", use_container_width=True):
            st.session_state.resume_data = {
                'personal_info': {'name': '', 'email': '', 'phone': '', 'linkedin': '', 'github': ''},
                'summary': '',
                'experience': [],
                'education': [],
                'skills': [],
                'projects': [],
                'certifications': []
            }
            st.success("✅ Form reset successfully!")
            st.rerun()
    
    # Resume completeness indicator
    st.markdown("---")
    completeness = calculate_resume_completeness(st.session_state.resume_data)
    st.subheader(f"📊 Resume Completeness: {completeness}%")
    st.progress(completeness / 100)

# FIXED DATE VALIDATION FUNCTION
def validate_date_format(date_str):
    """Validate date format MM/YYYY"""
    if date_str.lower() == 'present':
        return True
    try:
        parts = date_str.split('/')
        if len(parts) != 2:
            return False
        month, year = parts
        month_num = int(month)
        year_num = int(year)
        
        # Check if month is valid (1-12)
        if month_num < 1 or month_num > 12:
            return False
            
        # Check if year is reasonable (e.g., 1900-2100)
        if year_num < 1900 or year_num > 2100:
            return False
            
        return True
    except ValueError:
        return False

def calculate_resume_completeness(resume_data):
    """Calculate how complete the resume is"""
    total_points = 0
    earned_points = 0
    
    # Personal info (30 points)
    total_points += 30
    if resume_data['personal_info']['name']:
        earned_points += 10
    if resume_data['personal_info']['email']:
        earned_points += 10
    if resume_data['personal_info']['phone']:
        earned_points += 5
    if resume_data['personal_info']['linkedin']:
        earned_points += 5
    
    # Summary (15 points)
    total_points += 15
    if resume_data['summary']:
        earned_points += 15
    
    # Experience (25 points)
    total_points += 25
    if resume_data['experience']:
        earned_points += 25
    elif any([resume_data['personal_info']['name'], resume_data['personal_info']['email']]):
        earned_points += 10  # Partial credit if basic info exists
    
    # Education (15 points)
    total_points += 15
    if resume_data['education']:
        earned_points += 15
    
    # Skills (15 points)
    total_points += 15
    if resume_data['skills']:
        earned_points += 15
    
    return int((earned_points / total_points) * 100) if total_points > 0 else 0

# NEW: PDF GENERATION FUNCTION
def generate_pdf_resume(resume_data):
    """Generate a PDF resume from the resume data"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        import io
        
        # Create a bytes buffer for the PDF
        buffer = io.BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        story = []
        
        # Styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2c3e50')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=12,
            textColor=colors.HexColor('#34495e')
        )
        
        normal_style = styles['Normal']
        
        # Personal Information
        personal_info = resume_data['personal_info']
        name = personal_info.get('name', '')
        email = personal_info.get('email', '')
        phone = personal_info.get('phone', '')
        linkedin = personal_info.get('linkedin', '')
        
        # Add name as title
        if name:
            story.append(Paragraph(name.upper(), title_style))
        
        # Contact information
        contact_info = []
        if email:
            contact_info.append(f"Email: {email}")
        if phone:
            contact_info.append(f"Phone: {phone}")
        if linkedin:
            contact_info.append(f"LinkedIn: {linkedin}")
        
        if contact_info:
            story.append(Paragraph(" | ".join(contact_info), normal_style))
            story.append(Spacer(1, 20))
        
        # Professional Summary
        if resume_data.get('summary'):
            story.append(Paragraph("PROFESSIONAL SUMMARY", heading_style))
            story.append(Paragraph(resume_data['summary'], normal_style))
            story.append(Spacer(1, 12))
        
        # Work Experience
        if resume_data.get('experience'):
            story.append(Paragraph("WORK EXPERIENCE", heading_style))
            for exp in resume_data['experience']:
                position = exp.get('position', '')
                company = exp.get('company', '')
                start_date = exp.get('start_date', '')
                end_date = exp.get('end_date', '')
                description = exp.get('description', '')
                
                # Experience header
                exp_header = f"<b>{position}</b>"
                if company:
                    exp_header += f" | {company}"
                if start_date or end_date:
                    exp_header += f" | {start_date} - {end_date}"
                
                story.append(Paragraph(exp_header, normal_style))
                
                # Description
                if description:
                    # Clean up description for PDF
                    desc_lines = description.split('\n')
                    for line in desc_lines:
                        if line.strip():
                            story.append(Paragraph(f"• {line.strip()}", normal_style))
                
                story.append(Spacer(1, 8))
        
        # Education
        if resume_data.get('education'):
            story.append(Paragraph("EDUCATION", heading_style))
            for edu in resume_data['education']:
                degree = edu.get('degree', '')
                institution = edu.get('institution', '')
                start_date = edu.get('start_date', '')
                end_date = edu.get('end_date', '')
                description = edu.get('description', '')
                
                # Education header
                edu_header = f"<b>{degree}</b>"
                if institution:
                    edu_header += f" | {institution}"
                if start_date or end_date:
                    edu_header += f" | {start_date} - {end_date}"
                
                story.append(Paragraph(edu_header, normal_style))
                
                # Description
                if description:
                    story.append(Paragraph(description, normal_style))
                
                story.append(Spacer(1, 8))
        
        # Skills
        if resume_data.get('skills'):
            story.append(Paragraph("SKILLS", heading_style))
            skills_text = ", ".join(resume_data['skills'])
            story.append(Paragraph(skills_text, normal_style))
            story.append(Spacer(1, 12))
        
        # Projects
        if resume_data.get('projects'):
            story.append(Paragraph("PROJECTS", heading_style))
            for project in resume_data['projects']:
                name = project.get('name', '')
                technologies = project.get('technologies', '')
                description = project.get('description', '')
                link = project.get('link', '')
                
                # Project header
                proj_header = f"<b>{name}</b>"
                if technologies:
                    proj_header += f" | {technologies}"
                
                story.append(Paragraph(proj_header, normal_style))
                
                # Description
                if description:
                    story.append(Paragraph(description, normal_style))
                
                # Link
                if link:
                    story.append(Paragraph(f"Link: {link}", normal_style))
                
                story.append(Spacer(1, 8))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        return pdf_content
        
    except Exception as e:
        # Fallback: Return a simple text PDF
        st.warning(f"PDF generation issue: {str(e)}. Using fallback PDF.")
        return create_fallback_pdf(resume_data)

def create_fallback_pdf(resume_data):
    """Create a simple fallback PDF when advanced generation fails"""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    import io
    
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    # Set up basic info
    p.setFont("Helvetica-Bold", 16)
    personal_info = resume_data['personal_info']
    name = personal_info.get('name', 'Resume')
    p.drawString(100, 750, f"RESUME: {name.upper()}")
    
    y_position = 730
    
    # Contact info
    p.setFont("Helvetica", 10)
    contact_lines = []
    if personal_info.get('email'):
        contact_lines.append(f"Email: {personal_info['email']}")
    if personal_info.get('phone'):
        contact_lines.append(f"Phone: {personal_info['phone']}")
    
    for line in contact_lines:
        p.drawString(100, y_position, line)
        y_position -= 15
    
    y_position -= 10
    
    # Summary
    if resume_data.get('summary'):
        p.setFont("Helvetica-Bold", 12)
        p.drawString(100, y_position, "PROFESSIONAL SUMMARY")
        y_position -= 20
        p.setFont("Helvetica", 10)
        # Simple text wrapping
        summary = resume_data['summary']
        words = summary.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + word) < 80:
                current_line += word + " "
            else:
                lines.append(current_line)
                current_line = word + " "
        if current_line:
            lines.append(current_line)
        
        for line in lines:
            if y_position < 50:  # New page if needed
                p.showPage()
                y_position = 750
            p.drawString(100, y_position, line)
            y_position -= 15
        
        y_position -= 10
    
    p.save()
    pdf_content = buffer.getvalue()
    buffer.close()
    return pdf_content

# -----------------------------
# Career Analytics (Placeholder)
# -----------------------------
def run_career_analytics():
    """Career Analytics Dashboard"""
    st.header("📈 Career Analytics Dashboard")
    st.info("This feature is under development and will be available soon!")
    
    # Placeholder content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Market Trends")
        st.write("Coming soon: Real-time market demand analysis")
        
    with col2:
        st.subheader("Salary Insights")
        st.write("Coming soon: Industry salary benchmarks")

# -----------------------------
# Theme and Mobile Optimization
# -----------------------------
def mobile_optimized_layout():
    """Apply mobile-optimized CSS and layout"""
    st.markdown("""
    <style>
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        .stButton > button {
            width: 100%;
            margin: 5px 0;
        }
        .stTextInput > div > div > input {
            font-size: 16px;
        }
        .stSelectbox > div > div > select {
            font-size: 16px;
        }
        .card {
            margin: 10px 0;
            padding: 15px;
        }
        [data-testid="column"] {
            width: 100% !important;
        }
    }
    .stButton > button {
        min-height: 44px;
        min-width: 44px;
    }
    </style>
    """, unsafe_allow_html=True)

def apply_theme(mode_choice):
    if mode_choice == "Dark Mode":
        st.markdown("""
        <style>
        .main .block-container { background-color: #1a1d29; color: #e0e0e0; }
        .stApp { background-color: #1a1d29; }
        .stButton>button { background-color: #2d3746; color: #ffffff; border: 1px solid #3d4756; border-radius: 8px; }
        .stButton>button:hover { background-color: #3d4756; border: 1px solid #4d5766; }
        .stTextInput>div>div>input, .stSelectbox>div>div>select { background-color: #2d3746; color: #ffffff; border: 1px solid #3d4756; }
        .stSlider>div>div>div>div { background-color: #4d5766; }
        .stProgress>div>div>div>div { background: linear-gradient(90deg, #667eea, #764ba2); }
        a { color: #82c8ff !important; }
        a:hover { color: #a8d8ff !important; }
        .sidebar .sidebar-content { background-color: #2d3746; color: #e0e0e0; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .main .block-container { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #2c3e50; }
        .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .stButton>button { background: linear-gradient(45deg, #ffffff, #f8f9fa); color: #2c3e50; font-weight: bold; border: 2px solid #ffffff; border-radius: 8px; }
        .stButton>button:hover { background: linear-gradient(45deg, #f8f9fa, #ffffff); color: #2c3e50; border-color: #667eea; transform: translateY(-1px); }
        .stTextInput>div>div>input, .stSelectbox>div>div>select { background-color: rgba(255, 255, 255, 0.9) !important; color: #2c3e50; border: 2px solid rgba(255, 255, 255, 0.5); }
        .stSlider>div>div>div>div { background: linear-gradient(45deg, #667eea, #764ba2); }
        .stProgress>div>div>div>div { background: linear-gradient(90deg, #667eea, #764ba2); }
        a { color: #ffffff !important; background-color: rgba(255, 255, 255, 0.2); padding: 2px 6px; border-radius: 3px; }
        a:hover { color: #667eea !important; background-color: #ffffff; }
        .sidebar .sidebar-content { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #ffffff; }
        </style>
        """, unsafe_allow_html=True)

# -----------------------------
# Main App
# -----------------------------
def run():
    st.set_page_config(
        page_title="ResumeInsight Pro - AI Career Assistant", 
        page_icon='🎯', 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply mobile optimization
    mobile_optimized_layout()
    
    # Enhanced Dark/Light Mode
    mode_choice = st.sidebar.selectbox("Choose Mode:", ["Light Mode", "Dark Mode"])
    apply_theme(mode_choice)

    st.title("🚀 ResumeInsight Pro")
    st.markdown("### Your AI-Powered Career Growth Platform")
    
    # Enhanced Navigation with new features
    activities = [
        "User Dashboard", 
        "Resume Builder", 
        "Cover Letter Generator",
        "Resume Templates", 
        "Interview Questions",
        "Admin Panel", 
        "Career Analytics"
    ]
    
    choice = st.sidebar.selectbox("Navigate to:", activities)
    
    st.sidebar.markdown(
        '<div style="margin-top: 50px; padding: 15px; background-color: rgba(255,255,255,0.1); border-radius: 8px;">'
        '<p style="margin: 0; font-size: 14px; color: inherit;">© Developed by Arpit Sharma</p>'
        '<a href="https://www.linkedin.com/in/arpit-sharma-11b65b211/" style="color: inherit !important; font-size: 14px;">LinkedIn Profile</a>'
        '</div>', 
        unsafe_allow_html=True
    )

    if choice == "User Dashboard":
        run_user_section()
    elif choice == "Admin Panel":
        run_admin_section()
    elif choice == "Resume Builder":
        render_resume_builder()
    elif choice == "Cover Letter Generator":
        cover_letter_generator()
    elif choice == "Resume Templates":
        resume_templates()
    elif choice == "Interview Questions":
        interview_question_generator()
    elif choice == "Career Analytics":
        run_career_analytics()

if __name__ == "__main__":
    run()
