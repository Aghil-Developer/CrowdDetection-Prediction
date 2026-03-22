import streamlit as st
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression


st.set_page_config(
    page_title="Crowd Analytics | Enterprise Edition",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background-color: #0f172a;
    }
 
            
    div[data-testid="stMetricValue"] {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 14px !important;
    }
    
    .stMetric {
        background: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .status-active {
        color: #22c55e;
        font-weight: 600;
    }
    h1 {
        color: #f8fafc;
        letter-spacing: -0.025em;
        margin-bottom: 24px;
    }
    h3 {
        color: #94a3b8;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 16px;
    }
    </style>
    """, unsafe_allow_html=True)


with st.sidebar:
    st.title("Admin Console")
    st.markdown("---")
    
    run = st.checkbox("Enable Live Monitoring", value=False)
    st.divider()
    
    with st.expander("System Information", expanded=True):
        st.write("**Model:** YOLOv8 (Person Variant)")
        st.write("**Engine:** AI Inference Engine v2.4")
        st.write("**Region:** Global Monitoring")
    
    st.divider()
    st.markdown("### Support")
    st.caption("Developed by Aghil")


st.markdown("<h1>AI Enterprise Crowd Analytics Dashboard</h1>", unsafe_allow_html=True)


status_col1, status_col2, status_col3, status_col4 = st.columns(4)
with status_col1:
    st.caption("Status")
    st.markdown('<span class="status-active">● SYSTEM OPERATIONAL</span>', unsafe_allow_html=True)
with status_col2:
    st.caption("Inference Model")
    st.markdown("`YOLOv8-Native`")
with status_col3:
    st.caption("Processing Node")
    st.markdown("`LOCAL_HOST_01`")
with status_col4:
    st.caption("Uptime")
    uptime_placeholder = st.empty()

st.divider()



col1, col2 = st.columns([2.5, 1], gap="large")

with col1:
    st.markdown("1. Video Analytics Feed")
    FRAME_WINDOW = st.image([], use_container_width=True)
    
    
    alert_placeholder = st.empty()

with col2:
    st.markdown("2. Performance KPIs")
    
    
    kpi_col1, kpi_col2 = st.columns(2)
    with kpi_col1:
        count_metric = st.empty()
    with kpi_col2:
        density_metric = st.empty()
        
    prediction_metric = st.empty()
    
    st.markdown("3. Zonal Distribution")
    z_col1, z_col2, z_col3 = st.columns(3)
    with z_col1:
        zA_metric = st.empty()
    with z_col2:
        zB_metric = st.empty()
    with z_col3:
        zC_metric = st.empty()

    st.markdown("4. Trend Intelligence")
    graph_placeholder = st.empty()


model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

counts = []
times = []
densities = []
start_time = time.time()

def get_density(count):
    if count <= 5:
        return "LOW", (0, 255, 0)
    elif count <= 15:
        return "MEDIUM", (0, 255, 255)
    else:
        return "HIGH", (0, 0, 255)


while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not working")
        break

    h, w, _ = frame.shape
    
    
    zone_A = frame[:, :w//3]
    zone_B = frame[:, w//3:2*w//3]
    zone_C = frame[:, 2*w//3:]

    results = model(frame)

    people_count = 0
    heatmap = np.zeros((h, w), dtype=np.float32)
    zone_counts = {"A": 0, "B": 0, "C": 0}

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  
                people_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                heatmap[y1:y2, x1:x2] += 1

                
                center_x = (x1 + x2)//2
                if center_x < w//3:
                    zone_counts["A"] += 1
                elif center_x < 2*w//3:
                    zone_counts["B"] += 1
                else:
                    zone_counts["C"] += 1

    
    heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    
    density, color = get_density(people_count)
    current_time = time.time() - start_time
    counts.append(people_count)
    times.append(current_time)
    densities.append(density)

  
    pred_val = None
    if len(times) > 5:
        X = np.array(times).reshape(-1, 1)
        y = np.array(counts)
        model_lr = LinearRegression()
        model_lr.fit(X, y)
        future_time = np.array([[times[-1] + 5]])
        pred_val = int(model_lr.predict(future_time)[0])


 
    FRAME_WINDOW.image(overlay, channels="BGR")


    count_metric.metric("Total Count", f"{people_count}")
    density_metric.metric("Density", density)
    
    if pred_val is not None:
        prediction_metric.metric("Predicted (T+5s)", f"{pred_val} p/")
    
    
    zA_metric.metric("Zone A", f"{zone_counts['A']}")
    zB_metric.metric("Zone B", f"{zone_counts['B']}")
    zC_metric.metric("Zone C", f"{zone_counts['C']}")


    uptime_placeholder.markdown(f"`{int(current_time)}s`")
    
    
    if people_count > 20:
        alert_placeholder.warning("CRITICAL: Crowd Density is beyond optimal corporate thresholds!")
    else:
        alert_placeholder.empty()

    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#1e293b')
    
    
    ax.scatter(times, counts, c="#38bdf8", s=30, alpha=0.6, edgecolors="white", linewidth=0.5)
    ax.plot(times, counts, color="#38bdf8", alpha=0.4, linewidth=1.5)
    
    ax.set_xlabel("Time Progression (s)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Occupancy Count", color="#94a3b8", fontsize=9)
    ax.tick_params(axis='both', colors='#64748b', labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.1)
    
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    graph_placeholder.pyplot(fig)
    plt.close(fig)

cap.release()
