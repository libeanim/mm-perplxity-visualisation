import os
import json
import html
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image

import streamlit as st
import plotly.express as px

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


def get_available_runs(base_dir: str) -> List[str]:
    """
    List available run names based on JSON files in the recorder directory.
    """
    if not os.path.isdir(base_dir):
        return []
    
    json_files = [f for f in os.listdir(base_dir) if f.endswith(".json")]
    # Remove .json extension to get run names
    run_names = [os.path.splitext(f)[0] for f in json_files]
    return sorted(run_names)


def load_samples(
    base_dir: str,
    run_name: str,
) -> List[Dict[str, Any]]:
    """
    Load samples stored by SampleRecorder.

    SampleRecorder stores:
      - JSON file:  <base_dir>/<run_name>.json
      - Images in:  <base_dir>/<run_name>/*.jpg
    """
    json_path = os.path.join(base_dir, f"{run_name}.json")
    if not os.path.exists(json_path):
        st.error(f"JSON file not found: {json_path}")
        return []

    with open(json_path, "r") as f:
        samples = json.load(f)

    # Ensure consistent structure
    for s in samples:
        # Token count from input_ids
        input_ids = s.get("input_ids", [])
        if isinstance(input_ids, list):
            s["num_tokens"] = len(input_ids)
        else:
            s["num_tokens"] = 0

    return samples


def get_image_paths(
    sample: Dict[str, Any], base_dir: str, run_name: str
) -> List[str]:
    """
    Return all existing image paths for a sample.
    """
    img_keys = sample.get("images") or []
    img_paths: List[str] = []
    for key in img_keys:
        img_path = os.path.join(base_dir, run_name, key)
        if os.path.exists(img_path):
            img_paths.append(img_path)
    return img_paths


def render_tokens_with_mask(
    sample: Dict[str, Any],
    tokenizer,
):
    """
    Render tokens with mask highlighting (dark red for tokens contributing to loss).
    """
    input_ids = sample.get("input_ids") or []
    input_mask = sample.get("input_mask") or []
    if not input_ids or not input_mask:
        st.info("No input_ids/input_mask found for this sample.")
        return

    n = min(len(input_ids), len(input_mask))
    input_ids = input_ids[:n]
    input_mask = input_mask[:n]

    if tokenizer is not None:
        try:
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
        except Exception as e:
            st.warning(f"Failed to decode tokens with tokenizer: {e}")
            tokens = [str(t) for t in input_ids]
    else:
        tokens = [str(t) for t in input_ids]

    spans = []
    for tok, m in zip(tokens, input_mask):
        used = bool(m)
        color = "#b30000" if used else "#666666"
        weight = "700" if used else "400"
        # tokenizers use aspecial character to encode a leading space: "Ġ"
        # we replace it with a space
        vis_tok = tok.replace("Ġ", " ").replace("Ċ", "\\n")
        if vis_tok == "<|image_pad|>":
            continue
        spans.append(
            f'<span style="color:{color}; font-weight:{weight}; padding:1px;">{html.escape(vis_tok)}</span>'
        )
        if vis_tok == "<|vision_start|>":
            spans.append(
                f'<span style="color:{color}; font-weight:{weight}; padding:1px;">[image_pads hidden]</span>'
            )

    legend = (
        '<span style="color:#b30000; font-weight:700;">token used for loss</span>, '
        '<span style="color:#666666; font-weight:400;">token masked out</span>'
    )
    st.markdown(f"**Tokens (mask):** {legend}", unsafe_allow_html=True)
    st.markdown("".join(spans), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Multimodal Perplexity Viewer", layout="wide")
    st.title("Multimodal Perplexity Viewer")

    with st.sidebar:
        st.header("Data source")
        base_dir = st.text_input(
            "Recorder directory",
            value="TEST/recorder_juwels",
            help="Directory where SampleRecorder writes JSON + images.",
        )
        
        # Get available runs from JSON files in the directory
        available_runs = get_available_runs(base_dir)
        
        if available_runs:
            run_name = st.selectbox(
                "Run name",
                options=available_runs,
                help="Select a run based on available JSON files in the recorder directory.",
            )
        else:
            st.warning(f"No JSON files found in '{base_dir}'")
            run_name = st.text_input(
                "Run name (JSON prefix)",
                value="",
                help="No runs found. Enter a run name manually.",
            )
        tokenizer_name = st.text_input(
            "Tokenizer (for token view)",
            value="Qwen/Qwen2.5-VL-3B-Instruct",
            help="Hugging Face tokenizer used to decode input_ids.",
        )

        if st.button("Reload data"):
            st.cache_data.clear()
            st.cache_resource.clear()

    @st.cache_data(show_spinner=True)
    def _load_cached_samples(_base_dir: str, _run_name: str):
        return load_samples(_base_dir, _run_name)

    samples = _load_cached_samples(base_dir, run_name)

    if not samples:
        st.info("No samples loaded. Adjust the paths in the sidebar.")
        return

    @st.cache_resource(show_spinner=False)
    def _load_tokenizer(name: str):
        if AutoTokenizer is None:
            st.warning("transformers not installed; token view will show raw IDs.")
            return None
        try:
            return AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        except Exception as e:
            st.warning(f"Could not load tokenizer '{name}': {e}")
            return None

    tokenizer = _load_tokenizer(tokenizer_name)

    # Build dataframe-like structure for plotting
    cmi = []
    num_tokens = []
    indices = []
    types = []
    for i, s in enumerate(samples):
        cmi.append(s.get("cmi", np.nan))
        num_tokens.append(s.get("num_tokens", 0))
        indices.append(i)
        types.append(s.get("st", "unknown"))

    import pandas as pd

    df = pd.DataFrame(
        {
            "index": indices,
            "cmi": cmi,
            "num_tokens": num_tokens,
            "type": types,
        }
    )

    # Initialize session state for selected index
    if "selected_idx" not in st.session_state:
        st.session_state["selected_idx"] = 0

    # Layout: left = scatter, right = details
    col_plot, col_detail = st.columns([2, 1])

    with col_plot:
        st.subheader("CMI vs. Number of Tokens")

        scatter = px.scatter(
            df,
            x="num_tokens",
            y="cmi",
            color="type",
            hover_data=["index", "type", "num_tokens", "cmi"],
            custom_data=["index"],
        )
        scatter.update_traces(marker=dict(size=8, opacity=0.8))

        # Native Streamlit Plotly events (Streamlit >= 1.35)
        event_dict = st.plotly_chart(
            scatter,
            use_container_width=True,
            height=600,
            key="cmi_scatter",
            on_select="rerun",
            selection_mode="points",
        )
        # When the user clicks a point, update selected index from selection info
        if isinstance(event_dict, dict):
            sel = event_dict.get("selection") or {}
            points = sel.get("points") or []
            if points:
                # customdata[0] is our stored index
                cd = points[0].get("customdata") or []
                if cd:
                    clicked_idx = int(cd[0])
                    st.session_state["selected_idx"] = clicked_idx

    with col_detail:
        st.subheader("Selected sample")

        # Selector, driven by click but still manually overridable
        default_index = int(st.session_state.get("selected_idx", 0))
        selected_idx = st.number_input(
            "Sample index",
            min_value=0,
            max_value=len(samples) - 1,
            value=default_index,
            step=1,
            help="Click a point in the plot or adjust manually to inspect a sample.",
        )
        # Keep session state in sync with manual changes
        st.session_state["selected_idx"] = int(selected_idx)

        sample = samples[int(selected_idx)]

        st.markdown(f"**Index:** {int(selected_idx)}")
        st.markdown(f"**Type:** `{sample.get('st', 'unknown')}`")
        st.markdown(f"**CMI:** {sample.get('cmi')}")
        st.markdown(f"**CE(x):** {sample.get('ce_x')}")
        st.markdown(f"**CE(x,y):** {sample.get('ce_xy')}")
        st.markdown(f"**# tokens:** {len(sample.get('input_ids', []))}")

        # Get image paths for inline display in conversation
        img_paths = get_image_paths(sample, base_dir, run_name)

        # Show tokens with loss mask (collapsed by default)
        with st.expander("Tokens and loss mask", expanded=False):
            render_tokens_with_mask(sample, tokenizer)

        # Show conversation with inline images
        st.markdown("**Conversation:**")
        conv = sample.get("conversations", [])
        image_idx = 0  # Track which image to show next
        for turn in conv:
            speaker = turn.get("from", "unknown").upper()
            text = turn.get("value", "")
            
            # Split text by <image> placeholder and interleave with actual images
            parts = text.split("<image>")
            
            # Display first part with speaker label
            if parts[0].strip():
                st.markdown(f"**{speaker}:** {parts[0]}")
            else:
                st.markdown(f"**{speaker}:**")
            
            # For each subsequent part, show an image then the text
            for i, part in enumerate(parts[1:], start=1):
                # Show the next image if available
                if image_idx < len(img_paths):
                    try:
                        st.image(
                            Image.open(img_paths[image_idx]),
                            use_container_width=True,
                            caption=os.path.basename(img_paths[image_idx])
                        )
                    except Exception as e:
                        st.warning(f"Could not load image: {img_paths[image_idx]} ({e})")
                    image_idx += 1
                else:
                    st.warning(f"Missing image at position {image_idx}")
                    image_idx += 1
                
                # Show the text after this <image> placeholder
                if part.strip():
                    st.markdown(part)


if __name__ == "__main__":
    main()


