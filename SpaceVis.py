import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
import plotly.graph_objects as go
import requests
import os

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Prescriptive Maintenance Framework", layout="wide", page_icon="🔬")

st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    h1 { font-family: 'Times New Roman', serif; color: #2C3E50; }
    h2, h3 { font-family: 'Arial', sans-serif; color: #34495E; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_and_prep_data():
    data_url = "https://raw.githubusercontent.com/hankroark/cmapss/master/train_FD001.txt"
    file_name = "train_FD001.txt"

    if not os.path.exists(file_name):
        response = requests.get(data_url)
        with open(file_name, 'wb') as f:
            f.write(response.content)

    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f's_{i}' for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    df = pd.read_csv(file_name, sep='\s+', header=None, names=col_names)

    # RUL computation (run-to-failure)
    rul = df.groupby('unit_nr')['time_cycles'].max().reset_index()
    rul.columns = ['unit_nr', 'failure_cycle']
    df = df.merge(rul, on='unit_nr', how='left')
    df['RUL'] = df['failure_cycle'] - df['time_cycles']
    #After that we compute RUL, we can also delete the column (is not mandatory)
    df.drop(columns=['failure_cycle'], inplace=True)

    # RUL CAPPING (standard in C-MAPSS literature, I find this in a paper)
    #Early-life RUL is not meaningful and Prevents large RUL values dominating loss
    MAX_RUL = 125
    df['RUL'] = df['RUL'].clip(upper=MAX_RUL)

    return df

# --- 3. MODEL TRAINING & VALIDATION ---
#Prevents re-downloading and re-processing data every UI interaction
@st.cache_resource
def train_model(df):
    features = [
        's_2','s_3','s_4','s_7','s_8',
        's_11','s_12','s_13','s_14','s_15',
        's_17','s_20','s_21'
    ]

    X = df[features]
    y = df['RUL'] 
    groups = df['unit_nr']

    # Ensures that all cycles of one engine go entirely into either train or test set.
    splitter = GroupShuffleSplit(test_size=0.2, random_state=42) #random_state = 42 for reproducibility
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestRegressor(
        n_estimators=300, # number of trees
        max_depth=18, 
        min_samples_leaf=5,
        max_features='sqrt', #Number of features randomly considered at each split 
        random_state=42,
        n_jobs=-1 #use all CPU cores for faster training.
    )
    
    #Each tree learns patterns in the sensor data that predict RUL
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return model, features, rmse

# --- 4. VISUALIZATION FUNCTIONS ---
def plot_radar_chart(categories, original, modified):
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=original, theta=categories, fill='toself',
        name='Baseline State', line_color='gray'
    ))
    fig.add_trace(go.Scatterpolar(
        r=modified, theta=categories, fill='toself',
        name='Prescriptive State', line_color='blue'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title="Multivariate Sensor Deviation Analysis",
        showlegend=True,
        height=400
    )
    return fig

def plot_degradation_history(df, unit_id, current_cycle, sensor='s_2'):
    unit_data = df[df['unit_nr'] == unit_id]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=unit_data['time_cycles'],
        y=unit_data[sensor],
        mode='lines',
        name='History'
    ))

    current_val = unit_data.loc[
        unit_data['time_cycles'] == current_cycle, sensor
    ].values[0]

    fig.add_trace(go.Scatter(
        x=[current_cycle], y=[current_val],
        mode='markers', marker=dict(color='red', size=10),
        name='Current State'
    ))

    fig.update_layout(
        title=f"Historical Degradation Trajectory (Unit {unit_id})",
        xaxis_title="Operational Cycles",
        yaxis_title=f"Sensor Value ({sensor})",
        height=300
    )
    return fig

# --- 5. MAIN APPLICATION ---
st.title("Framework for Prescriptive Maintenance Optimization")
st.markdown("""
**Abstract:** Interactive digital twin for prescriptive maintenance using NASA C-MAPSS FD001.
A Random Forest regressor predicts Remaining Useful Life (RUL) and evaluates load-reduction strategies.
""")

df = load_and_prep_data()
model, feature_list, rmse_score = train_model(df)

# Sidebar
with st.sidebar:
    st.header("Experiment Setup")
    unit_id = st.selectbox("Select Engine", df['unit_nr'].unique(), index=9)
    unit_data = df[df['unit_nr'] == unit_id]
    max_cycle = int(unit_data['time_cycles'].max())
    #cycle sidebar creation (init the value at max_cycle*0.6)
    cycle_id = st.slider("Simulation Cycle", 1, max_cycle, int(max_cycle * 0.6))

    st.markdown("---")
    st.markdown(f"**RMSE:** `{rmse_score:.2f}` cycles")
    st.caption("≈ prediction uncertainty")

current_data = unit_data[unit_data['time_cycles'] == cycle_id].iloc[0]
input_data = pd.DataFrame([current_data[feature_list]])

col1, col2 = st.columns([1, 1.5])

with col1:
    #EVALUATION OF THE RUL AND NEW RUL
    prescriptive_factor = st.slider("Load Reduction Factor (α)", 0.9, 1.1, 1.0, 0.1)

    modified_input = input_data.copy()
    for s in ['s_2','s_3','s_4','s_7','s_8']:
        modified_input[s] *= prescriptive_factor

    base_rul = model.predict(input_data)[0]
    new_rul = model.predict(modified_input)[0]
    delta = new_rul - base_rul

    st.metric("Baseline RUL", f"{base_rul:.1f} cycles")
    st.metric("Prescribed RUL", f"{new_rul:.1f} cycles", f"{delta:+.1f}")


with col2:
    display_features = ['s_2','s_3','s_4','s_7','s_8']
    norm_orig = [1] * len(display_features)
    norm_mod = modified_input[display_features].iloc[0].values / input_data[display_features].iloc[0].values

    st.plotly_chart(
        plot_radar_chart(display_features, norm_orig, norm_mod),
        use_container_width=True
    )

st.divider()
st.subheader("Asset Degradation History")
st.plotly_chart(
    plot_degradation_history(df, unit_id, cycle_id),
    use_container_width=True
)

st.caption("Interactive prescriptive digital twin based on NASA C-MAPSS FD001.")


# RUL belongs to [0 125] hence we have a relative error equals to RMSE/125 = 17.21/125 = 13.8%
# My model predicts RUL with ~86% accuracy in time-to-failure terms.