import streamlit as st
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression
import sqlite3
import hashlib

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

def logout():
    st.session_state['authenticated'] = False
    st.rerun()


def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password_hash TEXT)''')
    conn.commit()
    conn.close()

def make_hash(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_username(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    data = c.fetchone()
    conn.close()
    return data

def add_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('INSERT INTO users(username, password_hash) VALUES (?,?)', 
              (username, make_hash(password)))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password_hash = ?', 
              (username, make_hash(password)))
    data = c.fetchone()
    conn.close()
    return data

init_db()

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

def logout():
    st.session_state['authenticated'] = False
    st.rerun()

st.set_page_config(
    page_title="Crowd Analytics | Enterprise Multi-Cam",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0f172a; }

/* Glassmorphism for containers */
[data-testid="stVerticalBlock"] > div:has(div.stForm) {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 40px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
}

div[data-testid="stMetricValue"] {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: #38bdf8 !important;
}

.stMetric {
    background: #1e293b;
    padding: 24px;
    border-radius: 12px;
    border: 1px solid #334155;
}

.status-active {
    color: #22c55e;
    font-weight: 600;
}

h1 {
    color: #f8fafc;
    margin-bottom: 24px;
    font-weight: 700;
}

.login-title {
    text-align: center;
    color: #38bdf8;
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 5px;
    letter-spacing: -1px;
}

.login-subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 16px;
    margin-bottom: 40px;
    font-weight: 400;
}

/* Center Tabs */
.stTabs [data-baseweb="tab-list"] {
    display: flex;
    justify-content: center;
    background-color: transparent;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.stTabs [data-baseweb="tab"] {
    font-weight: 600;
}

/* Form clean up */
div[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
}

/* Custom button styling */
.stButton>button {
    width: 100%;
    border-radius: 8px;
    height: 3em;
    background-color: #38bdf8;
    color: #0f172a;
    font-weight: 700;
    border: none;
    transition: all 0.3s;
}

.stButton>button:hover {
    background-color: #7dd3fc;
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
}
</style>
""", unsafe_allow_html=True)


if not st.session_state['authenticated']:
    _, center_col, _ = st.columns([1, 2, 1])
    
    with center_col:
        st.markdown('<div class="login-title">🛡️ CrowdAnalytics</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">Enterprise-Grade Multi-Camera Monitoring</div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Sign In", "Register"])

        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("Sign In")

                if submit:
                    if login_user(username, password):
                        st.session_state['authenticated'] = True
                        st.toast("Success! Access granted.", icon="🛡️")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Invalid credentials", icon="")

        with tab2:
            with st.form("register_form"):
                new_user = st.text_input("New Username", placeholder="Choose a username")
                new_pw = st.text_input("New Password", type="password", placeholder="At least 6 characters")
                confirm_pw = st.text_input("Confirm Password", type="password", placeholder="Repeat password")

                submit = st.form_submit_button("Create Account")

                if submit:
                    if new_pw != confirm_pw:
                        st.error("Passwords do not match", icon="")
                    elif check_username(new_user):
                        st.error("Username already exists", icon="")
                    elif len(new_pw) < 6:
                        st.error("Password must be at least 6 characters", icon="")
                    else:
                        add_user(new_user, new_pw)
                        st.success("Account created! You can now login.", icon="")
                        st.balloons()

    st.stop()


with st.sidebar:
    st.title("Admin Console")
    st.markdown("---")
    

    st.subheader("Camera Inventory")
    cam_input = st.text_area("Source List (Comma or space separated)", "0", help="Example: 0, 1 or 0 1 or rtsp://...")
    
    
    import re
    sources = re.split(r'[,\n\s]+', cam_input)
    sources = [s.strip() for s in sources if s.strip()]
    
    
    processed_sources = []
    for s in sources:
        try:
            processed_sources.append(int(s))
        except ValueError:
            processed_sources.append(s)
            
    run = st.checkbox("Enable Multi-Cam Engine", value=False)
    st.divider()
    
    density_threshold = st.slider("High Density Threshold", 5, 50, 20)
    
    with st.expander("System Specs"):
        st.write(f"**Loaded Sources:** {len(processed_sources)}")
        st.write("**Model:** YOLOv8-Native")
        st.write("**Inference:** Real-time Multi-threaded")
    
    st.divider()
    st.caption("Developed by Aghil | Enterprise v3.5")
    
    if st.button("Logout"):
        logout()


st.markdown("<h1>AI Multi-Camera Crowd Analytics Dashboard</h1>", unsafe_allow_html=True)


if not run:
    st.info("System on Standby. Configure camera sources in the sidebar and enable the engine to start monitoring.")
else:
    cols_per_row = 2
    rows = (len(processed_sources) + cols_per_row - 1) 
    
    cam_placeholders = []
    cam_containers = []
    
    
    for r in range(rows):
        st_cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            idx = r * cols_per_row + c
            if idx < len(processed_sources):
                with st_cols[c]:
                    cont = st.container()
                    cam_containers.append(cont)
                    cam_placeholders.append(cont.empty())
    
    st.divider()
    
    st.markdown("### System-Wide Intelligence")
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    total_count_metric = kpi_col1.empty()
    max_density_metric = kpi_col2.empty()
    system_status_metric = kpi_col3.empty()
    
    graph_placeholder = st.empty()


    @st.cache_resource
    def load_model():
        return YOLO("yolov8n.pt")

    model = load_model()
    
    
    history_counts = []
    history_times = []
    start_time = time.time()

    caps = []
    for src in processed_sources:
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            caps.append(cap)
        else:
            st.error(f"Failed to connect to source: {src}")

    try:
        while run:
            current_loop_counts = []
            
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret:
                    cam_placeholders[i].error(f"Loss of Signal: Cam {i+1}")
                    current_loop_counts.append(0)
                    continue

                h, w, _ = frame.shape
                
                results = model(frame, stream=True, verbose=False)
                people_count = 0
                
                for r in results:
                    for box in r.boxes:
                        if int(box.cls[0]) == 0:
                            people_count += 1
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                current_loop_counts.append(people_count)

                is_high = people_count >= density_threshold

                if is_high:
                    cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 10)

                    cam_placeholders[i].image(
                        frame,
                        channels="BGR",
                        use_container_width=True,
                        caption=f"🚨 CAMERA {i+1} | CRITICAL: {people_count} PEOPLE"
                    )
                else:
                    cam_placeholders[i].image(
                        frame,
                        channels="BGR",
                        use_container_width=True,
                        caption=f"CAMERA {i+1} | NORMAL: {people_count} PEOPLE"
                    )



            total_ppl = sum(current_loop_counts)
            max_ppl = max(current_loop_counts) if current_loop_counts else 0
            curr_time = time.time() - start_time
            
            history_counts.append(total_ppl)
            history_times.append(curr_time)

            total_count_metric.metric("Total Occupancy", f"{total_ppl} p")
            if caps:
                avg_val = total_ppl / len(caps)
                max_density_metric.metric("Max Unit Concentration", f"{max_ppl} p", delta=f"{max_ppl - avg_val:.1f} vs Avg")
            else:
                max_density_metric.metric("Max Unit Concentration", "N/A")
            system_status_metric.markdown(f'<span class="status-active">● ENGINE RUNNING | {int(curr_time)}s</span>', unsafe_allow_html=True)

            if len(history_times) % 5 == 0: 
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(10, 3))
                fig.patch.set_facecolor('#0f172a')
                ax.set_facecolor('#1e293b')
                
                ax.plot(history_times[-100:], history_counts[-100:], color="#38bdf8", linewidth=2)
                ax.fill_between(history_times[-100:], history_counts[-100:], alpha=0.1, color="#38bdf8")
                
                ax.set_title("System-Wide Occupancy Trend", color="#94a3b8", fontsize=10)
                ax.tick_params(axis='both', colors='#64748b', labelsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                graph_placeholder.pyplot(fig)
                plt.close(fig)

            time.sleep(0.01)

    finally:
        for cap in caps:
            cap.release()

