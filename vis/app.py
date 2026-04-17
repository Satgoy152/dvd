import sys
import os
import yaml
import time
import torch
import streamlit as st

# Add the project root to sys.path so we can import src/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.registry import ModelRegistry
from src.model import ModelBroker

st.set_page_config(page_title="MDLM Visualizer", layout="wide")

@st.cache_resource
def load_registry_and_names():
    registry_path = os.path.join(project_root, "registry.yaml")
    registry = ModelRegistry(registry_path)
    # Read the YAML to find available choices for the UI
    with open(registry_path, 'r') as f:
        conf = yaml.safe_load(f)
    models = [m['name'] for m in conf.get('available_models', [])]
    algos = [a['name'] for a in conf.get('available_algorithms', [])]
    return registry, models, algos

registry, available_models, available_algorithms = load_registry_and_names()

@st.cache_resource(show_spinner="Loading model into memory (this will take a while)...")
def get_model(model_name, role="VERIFIER"):
    model_module, model_config = registry.load_model(model_name)
    broker = ModelBroker(model_module, model_config, role=role)
    return broker

st.title("MDLM Token Generation Visualizer")
st.markdown("A simple UI to visualize step-by-step masked token decoding.")

with st.sidebar:
    st.header("Settings")
    
    default_model_idx = available_models.index("llada_8b_instruct") if "llada_8b_instruct" in available_models else 0
    model_choice = st.selectbox("Model", available_models, index=default_model_idx)
    
    default_algo_idx = available_algorithms.index("baseline_cascade") if "baseline_cascade" in available_algorithms else 0
    algo_choice = st.selectbox("Algorithm", available_algorithms, index=default_algo_idx)
    
    prompt = st.text_area("Prompt", value="Give me a short story for a 5 year old", height=100)
    steps = st.number_input("Steps", min_value=1, max_value=256, value=32, step=1)
    gen_len = st.number_input("Generation Length", min_value=1, max_value=2048, value=256, step=1)
    
    run_btn = st.button("Run Generation", type="primary")

if run_btn:
    if gen_len % steps != 0:
        st.error(f"Generation Length ({gen_len}) must be divisible by Steps ({steps}).")
        st.stop()

    st.markdown("---")
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    st.subheader("Decoding Visualization:")
    output_display = st.empty()
    
    # Render placeholder immediately
    status_text.info("Loading model...")
    
    # 1. Load model & algo
    try:
        verifier = get_model(model_choice)
        algo_func = registry.load_algorithm(algo_choice)
    except Exception as e:
        st.error(f"Failed to load model or algorithm: {e}")
        st.stop()
        
    tokenizer = verifier.model_state["tokenizer"]
    drafter = verifier # For baseline, Drafter == Verifier or is unused
    
    custom_kwargs = {
        "gen_length": int(gen_len),
        "mask_id": 126336,       # default mask id for LLaDA
        "tokenizer_max_len": 256 # similar to test_baseline.sh
    }
    algo_state = {}
    prompts = [prompt]
    
    status_text.info("Generation started...")
    
    for s in range(steps):
        # Update progress and status
        progress_bar.progress((s) / steps)
        status_text.info(f"Running step {s+1} / {steps}...")
        
        output = algo_func(
            drafter=drafter,
            verifier=verifier,
            prompts=prompts,
            step_idx=s,
            total_steps=steps,
            algo_state=algo_state,
            **custom_kwargs
        )
        
        algo_state = output.get("algo_state", algo_state)
        
        if "content" in algo_state and "prompt_len" in algo_state:
            full_content = algo_state["content"]
            prompt_len = algo_state["prompt_len"]
            
            gen_tokens = full_content[0, prompt_len:].tolist()
            
            # Format tokens as HTML
            html_parts = []
            for t in gen_tokens:
                if t == custom_kwargs["mask_id"]:
                    html_parts.append(
                        "<span style='background-color: #ffcccc; color: #cc0000; "
                        "padding: 1px 3px; margin: 1px; border-radius: 3px; "
                        "font-family: monospace; font-size: 0.85em;'>[MASK]</span>"
                    )
                else:
                    decoded = tokenizer.decode([t], skip_special_tokens=True)
                    if not decoded:
                        continue
                    # Escape HTML characters
                    decoded = decoded.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    decoded = decoded.replace("\n", "<br>")
                    html_parts.append(
                        f"<span style='background-color: #e6f2ff; color: #003366; "
                        f"padding: 1px 1px; margin: 0px;'>{decoded}</span>"
                    )
                    
            # Render html block
            output_display.markdown(
                f"<div style='line-height: 2.2; font-size: 1.1em;'>{''.join(html_parts)}</div>", 
                unsafe_allow_html=True
            )
            
        if output.get("done", False):
            progress_bar.progress(1.0)
            break
            
    status_text.success(f"Generation complete! ({steps} steps)")
    progress_bar.progress(1.0)
