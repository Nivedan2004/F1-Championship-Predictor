import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="F1 2025 – Final Standings Predictor", layout="wide")

# Initialize session state
if 'selected_race' not in st.session_state:
    st.session_state.selected_race = None
if 'selected_race_type' not in st.session_state:
    st.session_state.selected_race_type = None
if 'show_predictions' not in st.session_state:
    st.session_state.show_predictions = False
if 'championship_wins' not in st.session_state:
    st.session_state.championship_wins = None
if 'all_results' not in st.session_state:
    st.session_state.all_results = None

st.title("🏎️ F1 2025 – Championship Predictor")

# ------------------------
# Flag helpers
# ------------------------
FLAG_MAP = {
    "Bahrain": "🇧🇭", "Saudi Arabia": "🇸🇦", "Qatar": "🇶🇦", "Abu Dhabi": "🇦🇪",
    "Australia": "🇦🇺", "Singapore": "🇸🇬", "Japan": "🇯🇵", "China": "🇨🇳",
    "Spain": "🇪🇸", "Monaco": "🇲🇨", "Italy": "🇮🇹", "Emilia Romagna": "🇮🇹",
    "Austria": "🇦🇹", "Hungary": "🇭🇺", "Belgium": "🇧🇪", "Netherlands": "🇳🇱", 
    "Great Britain": "🇬🇧", "Azerbaijan": "🇦🇿",
    "USA": "🇺🇸", "USA Austin": "🇺🇸", "Miami": "🇺🇸", "Las Vegas": "🇺🇸", 
    "Mexico": "🇲🇽", "Brazil": "🇧🇷"
}

def flag_for_gp(gp: str) -> str:
    if gp in FLAG_MAP:
        return FLAG_MAP[gp]
    for k, v in FLAG_MAP.items():
        if k.lower() in gp.lower():
            return v
    return "🏁"

# ------------------------
# Sidebar
# ------------------------
with st.sidebar:
    st.header("Data")
    up_completed = st.file_uploader("completed_results.csv", type=["csv"])
    up_calendar  = st.file_uploader("remaining_calendar.csv", type=["csv"])
    up_track     = st.file_uploader("track_performance.csv", type=["csv"])

    st.header("Simulation")
    n_sims = st.slider("Monte Carlo simulations", 1000, 50000, 10000, step=1000)
    alpha = st.slider("Weight: Track pace", 0.1, 2.0, 1.0, 0.1)
    beta  = st.slider("Weight: Driver form", 0.1, 2.0, 0.8, 0.1)
    gamma = st.slider("Weight: Quali proxy", 0.0, 1.5, 0.4, 0.1)
    chaos = st.slider("Random race noise", 0.0, 2.5, 1.0, 0.1)
    seed  = st.number_input("Random seed", 42, step=1)

st.caption("See if your fav driver's taking the dub or if it's instant L this season 👀🏎️")

# ------------------------
# Load data
# ------------------------
def load_csv_or_demo(upload, fallback_path, demo_csv_str):
    if upload:
        return pd.read_csv(upload)
    try:
        return pd.read_csv(fallback_path)
    except Exception:
        return pd.read_csv(StringIO(demo_csv_str))

# Demo data
demo_completed_csv = """round,grand_prix,date,session,position,driver,team,points
1,Bahrain,2025-03-02,race,1,Oscar Piastri,McLaren,25
1,Bahrain,2025-03-02,race,2,Lando Norris,McLaren,18
1,Bahrain,2025-03-02,race,3,Max Verstappen,Red Bull,15
1,Bahrain,2025-03-02,race,4,George Russell,Mercedes,12
1,Bahrain,2025-03-02,race,5,Charles Leclerc,Ferrari,10
2,Saudi Arabia,2025-03-09,race,1,Oscar Piastri,McLaren,25
2,Saudi Arabia,2025-03-09,race,2,Lando Norris,McLaren,18
2,Saudi Arabia,2025-03-09,race,3,Max Verstappen,Red Bull,15
2,Saudi Arabia,2025-03-09,race,4,George Russell,Mercedes,12
2,Saudi Arabia,2025-03-09,race,5,Charles Leclerc,Ferrari,10
3,Australia,2025-03-23,race,1,Lando Norris,McLaren,25
3,Australia,2025-03-23,race,2,Oscar Piastri,McLaren,18
3,Australia,2025-03-23,race,3,Max Verstappen,Red Bull,15
3,Australia,2025-03-23,race,4,George Russell,Mercedes,12
3,Australia,2025-03-23,race,5,Charles Leclerc,Ferrari,10
"""

demo_calendar_csv = """round,grand_prix,date,has_sprint
16,Azerbaijan,2025-09-21,False
17,Singapore,2025-10-05,False
18,USA Austin,2025-10-19,True
19,Mexico,2025-11-02,False
20,Brazil,2025-11-16,True
21,Las Vegas,2025-11-23,False
22,Qatar,2025-12-07,False
23,Abu Dhabi,2025-12-14,False
"""

demo_track_csv = """grand_prix,team,pace_index,reliability
Bahrain,McLaren,1.08,0.97
Bahrain,Red Bull,1.06,0.98
Bahrain,Ferrari,1.02,0.97
Bahrain,Mercedes,1.01,0.96
Saudi Arabia,McLaren,1.07,0.97
Saudi Arabia,Red Bull,1.05,0.98
Saudi Arabia,Ferrari,1.01,0.97
Saudi Arabia,Mercedes,1.00,0.96
Australia,McLaren,1.07,0.97
Australia,Red Bull,1.04,0.98
Australia,Ferrari,1.01,0.97
Australia,Mercedes,1.00,0.96
"""

completed = load_csv_or_demo(up_completed, "data/completed_results.csv", demo_completed_csv)
calendar  = load_csv_or_demo(up_calendar,  "data/remaining_calendar.csv", demo_calendar_csv)
trackperf = load_csv_or_demo(up_track,     "data/track_performance.csv", demo_track_csv)

# Normalize dtypes
completed["date"] = pd.to_datetime(completed["date"], errors="coerce")
calendar["date"]  = pd.to_datetime(calendar["date"], errors="coerce")
completed["session"] = completed["session"].str.lower()
completed["driver"]  = completed["driver"].astype(str)
completed["team"]    = completed["team"].astype(str)

# Add flags
completed["flag"] = completed["grand_prix"].apply(flag_for_gp)
calendar["flag"]  = calendar["grand_prix"].apply(flag_for_gp)

# ------------------------
# Scoring tables
# ------------------------
RACE_POINTS = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}
SPRINT_POINTS = {1:8,2:7,3:6,4:5,5:4,6:3,7:2,8:1}

def score_points(df):
    dv = df.groupby(["driver","team"], as_index=False)["points"].sum()
    drv_totals = dv.groupby("driver", as_index=False)["points"].sum().rename(columns={"points":"driver_points"})
    teams = dv.groupby("team", as_index=False)["points"].sum().rename(columns={"points":"constructor_points"})
    return drv_totals.sort_values("driver_points", ascending=False), teams.sort_values("constructor_points", ascending=False)

def create_session_chart(session_data, title):
    """Create a beautiful session results chart"""
    if session_data.empty:
        st.info(f"No {title} data available")
        return
    
    # Sort by position
    session_data = session_data.sort_values("position")
    
    # Color mapping for teams
    team_colors = {
        'McLaren': '#FF8700',
        'Red Bull': '#3671C6', 
        'Ferrari': '#DC143C',
        'Mercedes': '#00D2BE',
        'Williams': '#005AFF',
        'Racing Bulls': '#3671C6',
        'Kick Sauber': '#52C41A',
        'Aston Martin': '#006F62',
        'Haas': '#FFFFFF',
        'Alpine': '#FF87BC'
    }
    
    # Create the chart
    fig = go.Figure()
    
    # Create horizontal bar chart with proper positioning
    positions = [f"P{pos}" for pos in session_data['position']]
    drivers = session_data['driver'].tolist()
    teams = session_data['team'].tolist()
    colors = [team_colors.get(team, '#808080') for team in teams]
    
    # Create individual bars for each driver to get correct hover data
    for i, (_, row) in enumerate(session_data.iterrows()):
        fig.add_trace(go.Bar(
            y=[positions[i]],
            x=[1],
            orientation='h',
            marker_color=colors[i],
            text=[drivers[i]],
            textposition='inside',
            textfont=dict(color='white', size=12),
            hovertemplate=f"<b>{row['driver']}</b><br>Team: {row['team']}<br>Position: P{row['position']}<extra></extra>",
            showlegend=False
        ))
    
    fig.update_layout(
        height=max(400, len(session_data) * 30),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=14)),
        margin=dict(l=0, r=0, t=20, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show results table
    st.markdown("**Results Table**")
    display_data = session_data[['position', 'driver', 'team']].copy()
    display_data.columns = ['Position', 'Driver', 'Team']
    st.dataframe(display_data, use_container_width=True, hide_index=True)

def show_race_details(race_name):
    """Show detailed race results with all sessions"""
    race_data = completed[completed["grand_prix"] == race_name].copy()
    
    # Create tabs for different sessions
    sessions = ["practice1", "practice2", "practice3", "qualifying", "race"]
    session_titles = ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"]
    
    # Filter to only show tabs that have data
    available_sessions = []
    available_titles = []
    for session, title in zip(sessions, session_titles):
        if not race_data[race_data["session"] == session].empty:
            available_sessions.append(session)
            available_titles.append(title)
    
    if available_sessions:
        tabs = st.tabs(available_titles)
        
        for i, (session, title) in enumerate(zip(available_sessions, available_titles)):
            with tabs[i]:
                session_data = race_data[race_data["session"] == session]
                create_session_chart(session_data, title)
    else:
        st.info("No session data available for this race.")

def show_main_page():
    """Show the main page with race selection and standings"""
    # Get unique races
    completed_races = completed[completed["session"].isin(["race", "sprint"])]["grand_prix"].unique()
    remaining_races = calendar["grand_prix"].unique()

    # Display completed races
    st.markdown("### ✅ Completed Races <small>(Click on the races to see the results)</small>", unsafe_allow_html=True)
    completed_cols = st.columns(5)

    for i, race in enumerate(completed_races):
        col_idx = i % 5
        with completed_cols[col_idx]:
            flag = flag_for_gp(race)
            if st.button(f"{flag} {race}", key=f"completed_{race}", use_container_width=True, type="primary"):
                st.session_state.selected_race = race
                st.session_state.selected_race_type = "completed"
                st.rerun()

    # Display remaining races
    st.markdown("### ⁉️ Remaining Races")
    remaining_cols = st.columns(4)

    for i, race in enumerate(remaining_races):
        col_idx = i % 4
        with remaining_cols[col_idx]:
            flag = flag_for_gp(race)
            if st.button(f"{flag} {race}", key=f"remaining_{race}", use_container_width=True, type="secondary"):
                st.session_state.selected_race = race
                st.session_state.selected_race_type = "remaining"
                st.rerun()

# ------------------------
    # Current Standings
# ------------------------
    st.markdown("---")
    st.subheader("📊 Current Championship Standings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Drivers Championship**")
        driver_standings = score_points(completed[completed["session"].isin(["race","sprint"])])[0]
    st.dataframe(
            driver_standings.rename(columns={"driver_points":"Points"}),
            use_container_width=True,
            hide_index=True
    )

    with col2:
        st.markdown("**Constructors Championship**")
        constructor_standings = score_points(completed[completed["session"].isin(["race","sprint"])])[1]
    st.dataframe(
            constructor_standings.rename(columns={"constructor_points":"Points"}),
            use_container_width=True,
            hide_index=True
    )


# ------------------------
# Prediction Functions
# ------------------------
def infer_trackpace_from_completed(completed_df, trackperf_df):
    """Infer track pace from completed race results"""
    race_data = completed_df[completed_df["session"] == "race"].copy()
    
    # Calculate inverse position (better position = higher score)
    race_data["inv_pos"] = 21 - race_data["position"]
    
    # Calculate pace index per team per track
    pace = race_data.groupby(["grand_prix", "team"]).agg({
        "inv_pos": "mean",
        "points": "sum"
    }).reset_index()
    
    # Normalize pace by track average
    pace["pace_index"] = pace.groupby("grand_prix")["inv_pos"].transform(lambda s: s / s.mean())
    
    # Merge with track performance data
    pace = pace.merge(trackperf_df, on=["grand_prix", "team"], how="left", suffixes=('', '_track'))
    
    # Use track performance data if available, otherwise use calculated pace_index
    pace["pace_index"] = pace["pace_index_track"].fillna(pace["pace_index"])
    pace["reliability"] = pace["reliability"].fillna(0.95)
    
    # Clean up extra columns
    pace = pace.drop(columns=["pace_index_track"], errors="ignore")
    
    return pace

def calculate_driver_form(completed_df):
    """Calculate recent driver form based on last 5 races"""
    race_data = completed_df[completed_df["session"] == "race"].copy()
    
    # Get driver points per race
    driver_points = race_data.groupby(["grand_prix", "driver"])["points"].sum().reset_index()
    
    # Calculate rolling average of last 5 races
    driver_form = {}
    for driver in driver_points["driver"].unique():
        driver_races = driver_points[driver_points["driver"] == driver].sort_values("grand_prix")
        if len(driver_races) >= 3:
            # Use last 3-5 races for form calculation
            recent_races = driver_races.tail(min(5, len(driver_races)))
            form = recent_races["points"].mean()
        else:
            form = driver_races["points"].mean() if len(driver_races) > 0 else 0
        
        driver_form[driver] = form
    
    return driver_form

def simulate_race(remaining_races, track_pace, driver_form, trackperf_df, completed_df, rng, alpha=1.0, beta=0.8, gamma=0.4, chaos=1.0):
    """Simulate a single race using Monte Carlo"""
    results = []
    
    for _, race in remaining_races.iterrows():
        gp_name = race["grand_prix"]
        has_sprint = race.get("has_sprint", False)
        
        # Get track pace for this race
        race_pace = track_pace[track_pace["grand_prix"] == gp_name]
        
        # If no track pace data, use average
        if race_pace.empty:
            race_pace = trackperf_df[trackperf_df["grand_prix"] == gp_name]
        
        if race_pace.empty:
            # Use default pace if no data
            teams = ["McLaren", "Red Bull", "Ferrari", "Mercedes", "Williams", "Racing Bulls", 
                    "Kick Sauber", "Aston Martin", "Haas", "Alpine"]
            race_pace = pd.DataFrame({
                "team": teams,
                "pace_index": [1.0] * len(teams),
                "reliability": [0.95] * len(teams)
            })
        
        # Calculate driver scores
        driver_scores = {}
        for _, team_data in race_pace.iterrows():
            team = team_data["team"]
            pace = team_data["pace_index"]
            reliability = team_data["reliability"]
            
            # Get drivers for this team
            team_drivers = completed_df[completed_df["team"] == team]["driver"].unique()
            
            for driver in team_drivers:
                form = driver_form.get(driver, 0)
                
                # Calculate base score
                base_score = alpha * pace + beta * (form / 25) + gamma * rng.normal(0, 0.1)
                
                # Add reliability factor
                if rng.random() > reliability:
                    base_score *= 0.5  # DNF penalty
                
                # Add chaos/noise
                noise = rng.normal(0, chaos * 0.2)
                final_score = base_score + noise
                
                driver_scores[driver] = {
                    "score": final_score,
                    "team": team,
                    "reliability": reliability
                }
        
        # Sort by score to get race positions
        sorted_drivers = sorted(driver_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # Assign race positions and points
        for pos, (driver, data) in enumerate(sorted_drivers, 1):
            points = RACE_POINTS.get(pos, 0)
            results.append({
                "grand_prix": gp_name,
                "driver": driver,
                "team": data["team"],
                "position": pos,
                "points": points,
                "session": "race"
            })
        
        # Simulate sprint if applicable
        if has_sprint:
            # Sprint positions (same order as race but different points)
            for pos, (driver, data) in enumerate(sorted_drivers, 1):
                points = SPRINT_POINTS.get(pos, 0)
                results.append({
                    "grand_prix": gp_name,
                    "driver": driver,
                    "team": data["team"],
                    "position": pos,
                    "points": points,
                    "session": "sprint"
                })
    
    return pd.DataFrame(results)

def run_monte_carlo_simulation(completed_df, remaining_races, trackperf_df, n_sims=10000, alpha=1.0, beta=0.8, gamma=0.4, chaos=1.0, seed=42):
    """Run Monte Carlo simulation to predict final standings"""
    rng = np.random.default_rng(seed)
    
    # Calculate track pace and driver form
    track_pace = infer_trackpace_from_completed(completed_df, trackperf_df)
    driver_form = calculate_driver_form(completed_df)
    
    # Run simulations
    all_results = []
    championship_wins = {"drivers": {}, "constructors": {}}
    
    for sim in range(n_sims):
        # Simulate remaining races
        simulated_results = simulate_race(remaining_races, track_pace, driver_form, trackperf_df, completed_df, rng, alpha, beta, gamma, chaos)
        
        # Combine with completed results
        combined_results = pd.concat([completed_df, simulated_results], ignore_index=True)
        
        # Calculate final standings
        race_data = combined_results[combined_results["session"].isin(["race", "sprint"])]
        driver_standings, constructor_standings = score_points(race_data)
        
        # Track championship winners
        driver_champ = driver_standings.iloc[0]["driver"]
        constructor_champ = constructor_standings.iloc[0]["team"]
        
        championship_wins["drivers"][driver_champ] = championship_wins["drivers"].get(driver_champ, 0) + 1
        championship_wins["constructors"][constructor_champ] = championship_wins["constructors"].get(constructor_champ, 0) + 1
        
        # Store results for analysis
        all_results.append({
            "simulation": sim,
            "driver_champion": driver_champ,
            "constructor_champion": constructor_champ,
            "driver_standings": driver_standings,
            "constructor_standings": constructor_standings
        })
    
    return championship_wins, all_results


# ------------------------
# Main App Logic
# ------------------------
# Show selected race details
if 'selected_race' in st.session_state and st.session_state.selected_race:
    selected_race = st.session_state.selected_race
    race_type = st.session_state.selected_race_type
    
    # Back button
    if st.button("← Back to Race Selection", key="back_button"):
        st.session_state.selected_race = None
        st.rerun()
    
    st.markdown("---")
    st.subheader(f"🏎️ {selected_race} Grand Prix Results")
    
    if race_type == "completed":
        show_race_details(selected_race)
    else:
        st.info(f"🏁 {selected_race} Grand Prix is yet to be held. Check back after the race weekend!")
else:
    show_main_page()



# ------------------------
# Championship Predictions
# ------------------------
st.markdown("---")
st.subheader("🔮 Championship Predictions")

# Simple run button
if st.button("🎲 Run Championship Prediction", type="primary", use_container_width=True):
    try:
        with st.spinner("Running Monte Carlo simulation... This may take a moment."):
            # Reduce simulation count for faster testing
            test_sims = min(n_sims, 1000)  # Cap at 1000 for faster results
            
            # Run the simulation
            championship_wins, all_results = run_monte_carlo_simulation(
                completed, calendar, trackperf, test_sims, alpha, beta, gamma, chaos, seed
            )
            
            # Store results in session state
            st.session_state.championship_wins = championship_wins
            st.session_state.all_results = all_results
            st.session_state.show_predictions = True
            st.success(f"✅ Simulation completed! Ran {test_sims:,} simulations.")
            st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error running simulation: {str(e)}")
        st.info("Please check the console for more details.")

# Show predictions page if simulation completed
if st.session_state.get("show_predictions", False) and st.session_state.championship_wins:
    championship_wins = st.session_state.championship_wins
    all_results = st.session_state.all_results
    
    # Calculate probabilities
    total_sims = len(all_results)
    
    # Driver championship probabilities
    driver_probs = {driver: (wins / total_sims) * 100 
                   for driver, wins in championship_wins["drivers"].items()}
    driver_probs = dict(sorted(driver_probs.items(), key=lambda x: x[1], reverse=True))
    
    # Ensure we have at least top 3 drivers (add 0% for missing ones)
    all_drivers = set(completed["driver"].unique())
    for driver in all_drivers:
        if driver not in driver_probs:
            driver_probs[driver] = 0.0
    driver_probs = dict(sorted(driver_probs.items(), key=lambda x: x[1], reverse=True))
    
    # Constructor championship probabilities  
    constructor_probs = {team: (wins / total_sims) * 100 
                       for team, wins in championship_wins["constructors"].items()}
    constructor_probs = dict(sorted(constructor_probs.items(), key=lambda x: x[1], reverse=True))
    
    # Ensure we have at least top 3 teams (add 0% for missing ones)
    all_teams = set(completed["team"].unique())
    for team in all_teams:
        if team not in constructor_probs:
            constructor_probs[team] = 0.0
    constructor_probs = dict(sorted(constructor_probs.items(), key=lambda x: x[1], reverse=True))
    
    # Back button
    if st.button("← Back to Main Page", key="back_to_main"):
        st.session_state.show_predictions = False
        st.rerun()
    
    st.title("🏆 2025 F1 Championship Predictions")
    st.markdown("---")
    
    # Display champions prominently
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏎️ Driver Champion")
        if driver_probs:
            most_likely_driver = max(driver_probs, key=driver_probs.get)
            prob = driver_probs[most_likely_driver]
            
            # Large display for champion
            st.markdown(f"# {most_likely_driver}")
            st.markdown(f"## {prob:.1f}% Probability")
    
    with col2:
        st.markdown("### 🏆 Constructor Champion")
        if constructor_probs:
            most_likely_team = max(constructor_probs, key=constructor_probs.get)
            prob = constructor_probs[most_likely_team]
            
            # Large display for champion
            st.markdown(f"# {most_likely_team}")
            st.markdown(f"## {prob:.1f}% Probability")
    
    # Simulation details
    st.markdown("---")
    st.markdown("### 📊 Simulation Details")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Simulations", f"{total_sims:,}")
    with col2:
        st.metric("Remaining Races", len(calendar))
    with col3:
        st.metric("Sprint Races", len(calendar[calendar.get("has_sprint", False) == True]))
    with col4:
        st.metric("Current Leader", list(driver_probs.keys())[0] if driver_probs else "N/A")
    
    
    # Add the information section
    st.markdown("---")
    st.markdown("### ℹ️ How the Prediction Works")
    
    st.markdown("""
    **How it works:**
    1. **Analyzes completed races** to determine each team's pace at different tracks
    2. **Calculates driver form** based on recent performance (last 5 races)
    3. **Simulates remaining races** using Monte Carlo method with realistic randomness
    4. **Includes sprint races** where applicable
    5. **Runs thousands of simulations** to calculate championship probabilities
    
    **Adjust the parameters in the sidebar to see how different factors affect predictions:**
    - **Track Pace Weight**: How much track-specific performance matters
    - **Driver Form Weight**: How much recent driver performance matters  
    - **Qualifying Proxy**: How much qualifying position affects race outcome
    - **Random Race Noise**: How much randomness/unpredictability to include
    """)
    
    # Show current simulation parameters
    st.markdown("### ⚙️ Current Simulation Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Track Pace Weight", f"{alpha:.1f}")
    with col2:
        st.metric("Driver Form Weight", f"{beta:.1f}")
    with col3:
        st.metric("Qualifying Proxy", f"{gamma:.1f}")
    with col4:
        st.metric("Random Race Noise", f"{chaos:.1f}")
    
    st.info("💡 To change these parameters, go back to the main page and adjust the sliders in the sidebar, then run the simulation again.")

else:
    st.info("👆 Click the button above to run the championship prediction simulation!")
    st.markdown("""
    **How it works:**
    1. **Analyzes completed races** to determine each team's pace at different tracks
    2. **Calculates driver form** based on recent performance (last 5 races)
    3. **Simulates remaining races** using Monte Carlo method with realistic randomness
    4. **Includes sprint races** where applicable
    5. **Runs thousands of simulations** to calculate championship probabilities
    
    **Adjust the parameters in the sidebar to see how different factors affect predictions:**
    - **Track Pace Weight**: How much track-specific performance matters
    - **Driver Form Weight**: How much recent driver performance matters  
    - **Qualifying Proxy**: How much qualifying position affects race outcome
    - **Random Race Noise**: How much randomness/unpredictability to include
    """)
st.caption("Method: current results → standings; estimate driver form + per-track team pace/reliability; simulate remaining GPs with sprint rules; aggregate many runs to get title odds.")
