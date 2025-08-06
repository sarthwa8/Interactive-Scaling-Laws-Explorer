# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go  
import scaling_logic as sl

# --- Page Configuration ---
st.set_page_config(
    page_title="Scaling Laws Explorer",
    page_icon="🔬",
    layout="wide"
)

# --- Data Loading (with caching to improve performance) ---
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath, header=None, names=['N', 'Loss'])

# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a Page", ["Introduction", "Core Scaling Laws", "The Overfitting Frontier", "The Compute-Efficient Frontier"])

# ==============================================================================
# PAGE 1: INTRODUCTION
# ==============================================================================

if page == "Introduction":
    st.title(" The Interactive Scaling Laws Explorer")
    st.markdown("""
    Welcome! This dashboard is an interactive tool to explore the findings of the landmark paper **"Scaling Laws for Neural Language Models"** by Kaplan et al. (OpenAI, 2020).
    
    Since replicating the experiments requires massive computational resources, this app focuses on **verifying and visualizing** the paper's key conclusions using their published data and mathematical models.
    
    Use the navigation sidebar to explore different aspects of the scaling laws.
    
    - **Original Paper:** [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
    - **Dashboard Source Code:** [GitHub](https://github.com/) (Link to your repo)
    """)

# ==============================================================================
# PAGE 2: CORE SCALING LAWS (FINALIZED WITH USER'S DATASET NAMES)
# ==============================================================================
elif page == "Core Scaling Laws":
    st.header("The Core Power Laws: A Deeper Look at Architecture")
    st.markdown("""
    This section visualizes the data digitized from **Figure 6 (Right)** of the paper.
    
    The paper's key insight here is that for models with more than 2 layers, performance converges to a single power-law trend. However, very shallow models (1-2 layers) perform worse for a given parameter count. Use the checkboxes below to explore this phenomenon.
    """)
    
    # --- Load the architectural data ---
    @st.cache_data
    def load_architectural_data(filepath):
        return pd.read_csv(filepath)

    try:
        df_arch = load_architectural_data('architectural_data.csv')
        
        # --- Interactive Controls in Columns ---
        st.subheader("Display Options")
        st.write("Toggle the visibility of each architectural series from the paper:")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            show_1_layer = st.checkbox("1 Layer", value=True)
        with col2:
            show_2_layers = st.checkbox("2 Layers", value=True)
        with col3:
            show_3_layers = st.checkbox("3 Layers", value=True)
        with col4:
            show_6_layers = st.checkbox("6 Layers", value=True)
        with col5:
            show_deep_models = st.checkbox("More than 6 Layers", value=True)

        # Theoretical curve slider
        alpha_n_slider = st.slider(
            "Theoretical Scaling Exponent (α_N)", 
            min_value=0.05, max_value=0.1, value=sl.ALPHA_N, step=0.001, format="%.3f"
        )
        show_theory = st.checkbox("Show Theoretical Curve", value=True)

        # --- Dynamic Plotting with Plotly (USING CORRECT COLUMN NAMES) ---
        fig = go.Figure()

        # Add theoretical curve
        if show_theory:
            n_range = np.logspace(3, 9.5, 100)
            loss_theoretical = sl.calculate_loss_vs_n(n_range, alpha_n_slider)
            fig.add_trace(go.Scatter(x=n_range, y=loss_theoretical, mode='lines', name=f'Theory (α_N={alpha_n_slider:.3f})', line=dict(dash='dash', color='grey')))

        # Conditionally add digitized data using the CORRECT column names from your file
        if show_1_layer and '1_layer' in df_arch.columns:
            fig.add_trace(go.Scatter(x=df_arch['1_layer'].dropna(), y=df_arch['Unnamed: 1'].dropna(), mode='markers', name='1 Layer'))
        if show_2_layers and '2_layers' in df_arch.columns:
            fig.add_trace(go.Scatter(x=df_arch['2_layers'].dropna(), y=df_arch['Unnamed: 3'].dropna(), mode='markers', name='2 Layers'))
        if show_3_layers and '3_layers' in df_arch.columns:
            fig.add_trace(go.Scatter(x=df_arch['3_layers'].dropna(), y=df_arch['Unnamed: 5'].dropna(), mode='markers', name='3 Layers'))
        if show_6_layers and '6_layers' in df_arch.columns:
            fig.add_trace(go.Scatter(x=df_arch['6_layers'].dropna(), y=df_arch['Unnamed: 7'].dropna(), mode='markers', name='6 Layers'))
        if show_deep_models and 'morethan6_layers' in df_arch.columns:
            fig.add_trace(go.Scatter(x=df_arch['morethan6_layers'].dropna(), y=df_arch['Unnamed: 9'].dropna(), mode='markers', name='> 6 Layers'))
            
        # Update layout
        fig.update_layout(
            title="Test Loss vs. Parameters by Model Depth",
            xaxis_title="Parameters (non-embedding)",
            yaxis_title="Test Loss",
            xaxis_type="log",
            yaxis_type="log",
            legend_title="Model Depth"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(
            """
            **Analysis:** As you toggle the checkboxes, notice how the data for models with 3 or more layers falls along the same theoretical trend line.
            In contrast, the 1 and 2-layer models are consistently above the line, indicating worse performance for their size.
            This visually confirms the paper's conclusion that performance depends strongly on scale (total parameters), but only weakly on shape (like depth), provided the model is sufficiently deep.
            """
        )

    except FileNotFoundError:
        st.error("Error: `architectural_data.csv` not found. Please make sure you have saved the digitized data from Figure 6 and named the file correctly.")
    except Exception as e:
        st.error(f"An error occurred: {e}. Please check that your CSV column names match what is in the code.")

# ==============================================================================
# PAGE 3: OVERFITTING FRONTIER
# ==============================================================================
elif page == "The Overfitting Frontier":
    st.header("The Overfitting Frontier: Model Size vs. Dataset Size")
    st.markdown("Explore the trade-off between model size (N) and dataset size (D) using Equation 1.5 from the paper[cite: 204].")
    
    col1, col2 = st.columns(2)
    with col1:
        n_slider = st.slider("Model Size (N)", min_value=1e6, max_value=1e11, value=1e9, format="%.0e")
    with col2:
        d_slider = st.slider("Dataset Size (D)", min_value=1e8, max_value=1e13, value=2.2e10, format="%.0e")
        
    # Calculate loss and status
    predicted_loss, status = sl.calculate_overfitting_loss(n_slider, d_slider)
    
    st.metric(label="Predicted Test Loss", value=f"{predicted_loss:.3f}")
    st.subheader("Verdict:")
    st.info(status)
    st.markdown("""
    This shows which factor is limiting performance. When **overfitting**, you need more data. When **model capacity is the bottleneck**, you need a larger model.
    """)

# ==============================================================================
# PAGE 4: COMPUTE-EFFICIENT FRONTIER
# ==============================================================================
elif page == "The Compute-Efficient Frontier":
    st.header("The Compute-Efficient Frontier")
    st.markdown("What's the best way to allocate a fixed compute budget? This calculator uses the paper's findings (Figure 14) to determine the optimal model size and training steps for a given amount of compute.")
    
    compute_budget = st.select_slider(
        "Compute Budget C_min (in Petaflop-days)",
        options=np.logspace(-7, 2, 100),
        value=1e-3,
        format_func=lambda x: f"{x:.2e} PF-days"
    )

    # Calculate optimal values using the backend logic
    n_opt = sl.calculate_optimal_model_size(compute_budget)
    s_opt = sl.calculate_optimal_steps(compute_budget)
    l_best = sl.calculate_best_loss_from_compute(compute_budget)

    st.subheader("Optimal Allocation for Your Budget:")
    col1, col2, col3 = st.columns(3)
    col1.metric("Optimal Model Size (N_opt)", f"{n_opt/1e9:.3f} B Params")
    col2.metric("Optimal Training Steps (S_min)", f"{s_opt:,.0f} Steps")
    col3.metric("Best Possible Test Loss (L)", f"{l_best:.3f}")

    st.warning("**Key Insight:** The paper shows that for optimal training, you should use most of your extra compute budget on **making the model larger**, not on training for more steps[cite: 95, 760].")