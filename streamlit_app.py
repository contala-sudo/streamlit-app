import os
import time
import random
import json
import streamlit as st
import pandas as pd
from openai import OpenAI
# Fine-tuned model mappings
MODEL_MAPPING = {
    "ART-6k": "ft:gpt-4.1-mini-2025-04-14:contentwhale:cw-art-gen-fin-6k:D06XDIUh",
    "ART-10k": "ft:gpt-4.1-mini-2025-04-14:contentwhale:art-finance-rstv2-4contenttype-10k-2:Dbz18jqM"
}

# Prompt template from original script
ARTICLE_GENERATION_PROMPT = """You are an expert professional writer. Given an article brief and its rhetoric structure, generate an article.
Brief:
Title: {article_title}
Type of Article: {content_type}
Word Count: {word_count}
Tone: {tone}
Target Geography: {geographic_target}
Target Demography: {demographic_target}
Primary Keywords: {primary_keywords}
Client Name: {client_name}
Client Description: {client_description}
Client Industry: {industry}
Purpose/Objective: {purpose_objective}
Additional Information: {additional_information}
---
Rhetoric Structure: {rhetoric_structure}"""

# Set page config for professional design
st.set_page_config(
    page_title="Article Generator - Model Testing",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
    <style>
        /* Modern font and colors */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Outfit', sans-serif;
            background-color: #0b0f19;
            color: #f3f4f6;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #0e1322 !important;
            border-right: 1px solid #1e293b;
        }
        
        /* Header titles */
        h1, h2, h3 {
            color: #ffffff;
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        
        /* Glassmorphism containers */
        .glass-card {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
        }
        
        /* Stat/Metric Card styling */
        .metric-container {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            flex: 1;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            color: #3b82f6;
            margin-bottom: 4px;
        }
        
        .metric-label {
            font-size: 12px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Button styling */
        div.stButton > button {
            background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            border: none !important;
            padding: 12px 30px !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
            transition: all 0.2s ease-in-out !important;
            width: 100% !important;
        }
        
        div.stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5) !important;
            background: linear-gradient(135deg, #60a5fa, #2563eb) !important;
        }
        
        /* Download button styling */
        [data-testid="stDownloadButton"] > button {
            background: linear-gradient(135deg, #10b981, #047857) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            border: none !important;
            padding: 12px 30px !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
            transition: all 0.2s ease-in-out !important;
            width: 100% !important;
        }
        
        [data-testid="stDownloadButton"] > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
            background: linear-gradient(135deg, #34d399, #059669) !important;
        }
    </style>
""", unsafe_allow_html=True)

# Build the prompt matching SQL behavior
def get_prompt_for_row(row):
    return ARTICLE_GENERATION_PROMPT.format(
        article_title=row.get('article_title') or "N/A",
        content_type=row.get('content_type_ai') or "N/A",
        word_count=row.get('article_word_count') or "N/A",
        tone=row.get('tone_ai') or "N/A",
        geographic_target=row.get('geographic_target_ai') or "N/A",
        demographic_target=row.get('demographic_target_ai') or "N/A",
        primary_keywords=row.get('primary_keywords_ai') or "N/A",
        client_name=row.get('client_name') or "N/A",
        client_description=row.get('client_description_ai') or "N/A",
        industry=row.get('industry_ai') or row.get('brief_industry') or "N/A",
        purpose_objective=row.get('brief_purpose_objective') or "N/A",
        additional_information=row.get('brief_additional_information') or "N/A",
        rhetoric_structure=row.get('rhetoric_structure_actual') or "N/A"
    )

# OpenAI call execution logic with retries
def call_openai_model(api_key, model_id, prompt, temp, top_p, freq_penalty, max_retries=3):
    client = OpenAI(api_key=api_key)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                max_completion_tokens=16384,
                temperature=temp,
                top_p=top_p,
                frequency_penalty=freq_penalty,
                store=False,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]
            )
            
            if resp is None or not hasattr(resp, 'choices') or not resp.choices:
                return None, 0, 0, 0, "Empty response from OpenAI API"

            content = resp.choices[0].message.content
            usage = resp.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0
            
            return content, prompt_tokens, completion_tokens, total_tokens, None

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                return None, 0, 0, 0, str(e)

def reset_db_state():
    for key in ["db_generated_content", "db_prompt_tokens", "db_completion_tokens", "db_total_tokens", "db_elapsed_time", "db_generated_for_id", "db_select_brief"]:
        if key in st.session_state:
            del st.session_state[key]

def reset_custom_state():
    for key in ["custom_generated_content", "custom_prompt_tokens", "custom_completion_tokens", "custom_total_tokens", "custom_elapsed_time"]:
        if key in st.session_state:
            del st.session_state[key]
    for key in ["cust_title", "cust_content_type", "cust_word_count", "cust_tone", "cust_geo", "cust_demography", "cust_keywords", "cust_client_name", "cust_client_desc", "cust_industry", "cust_purpose", "cust_additional_info", "cust_rhetoric"]:
        if key in st.session_state:
            del st.session_state[key]

def reset_all_state():
    reset_db_state()
    reset_custom_state()

def main():
    # Sidebar parameter configuration
    # API Key check from Streamlit secrets
    api_key = None
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.sidebar.info("API key found in secrets.")
    else:
        st.sidebar.error("⚠️ OPENAI_API_KEY not found in Streamlit secrets!")
    
    st.sidebar.markdown("## ⚙️ Model Parameters")
    
    selected_model_name = st.sidebar.selectbox(
        "Choose Fine-Tuned Model",
        options=["ART-6k", "ART-10k"],
        index=0,
        on_change=reset_all_state
    )
    model_id = MODEL_MAPPING[selected_model_name]
    
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.2, step=0.1, on_change=reset_all_state)
    st.sidebar.info("High - More creative, less predictable")
    top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.05, on_change=reset_all_state)
    st.sidebar.info("High - more words for the model to pick from")
    frequency_penalty = st.sidebar.slider("Frequency Penalty", min_value=0.0, max_value=2.0, value=0.0, step=0.1, on_change=reset_all_state)
    st.sidebar.info("High - more penalty for repeated words")

    st.markdown("# 📝 Fine-Tuned Article Generator")
    st.markdown("Convert article briefs into high-quality articles using fine-tuned OpenAI models.")

    tab1, tab2 = st.tabs(["📋 Predefined Briefs (Database)", "✍️ Custom Brief & Rhetoric"])

    with tab1:
        # Load local JSON file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, "articles_data.json")

        if not os.path.exists(json_path):
            st.error(f"❌ Data file not found at `{json_path}`. Please run `python export_data.py` to dump data first.")
        else:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
            except Exception as e:
                st.error(f"❌ Failed to parse `articles_data.json`: {e}")
                articles = None

            if articles:
                # Convert to DataFrame to display to the user in tabular format
                df = pd.DataFrame(articles)

                # Re-order columns slightly to show key details first
                col_order = [
                    'article_title', 'content_type_ai', 'article_word_count', 
                    'tone_ai', 'geographic_target_ai', 'demographic_target_ai', 'primary_keywords_ai',
                    'client_name', 'client_description_ai', 'industry_ai', 'brief_industry', 
                    'brief_purpose_objective', 'brief_additional_information', 'rhetoric_structure_actual'
                ]
                # Filter to only keep columns that actually exist
                col_order = [c for c in col_order if c in df.columns]
                df_display = df[col_order]

                st.markdown("### 📋 Available Article Briefs")
                st.dataframe(df_display, use_container_width=True, hide_index=True)

                # Allow user to pick one
                st.markdown("### 🎯 Select Target Brief")
                selected_id = st.selectbox(
                    "Choose an article brief to generate:",
                    options=df['google_doc_id'].tolist(),
                    format_func=lambda x: df[df['google_doc_id'] == x]['article_title'].values[0],
                    on_change=reset_db_state,
                    key="db_select_brief"
                )

                # Retrieve selected row
                selected_row = next(item for item in articles if item["google_doc_id"] == selected_id)

                # Show brief selection summary
                st.markdown(f"""
                    <div class="glass-card">
                        <h4>Selected: {selected_row['article_title']}</h4>
                        <span style="background-color:#1e3a8a; padding:4px 8px; border-radius:4px; font-size:12px; margin-right:6px;">Type: {selected_row['content_type_ai'] or 'N/A'}</span>
                        <span style="background-color:#14532d; padding:4px 8px; border-radius:4px; font-size:12px; margin-right:6px;">Word Count: {selected_row['article_word_count'] or 'N/A'}</span>
                        <span style="background-color:#581c87; padding:4px 8px; border-radius:4px; font-size:12px; margin-right:6px;">Tone: {selected_row['tone_ai'] or 'N/A'}</span>
                    </div>
                """, unsafe_allow_html=True)

                # Generation Button and execution
                if st.button("🚀 Generate Article", key="btn_db_generate"):
                    if not api_key:
                        st.error("API Key is missing. Please add `OPENAI_API_KEY` to your environment or `.env` file.")
                    else:
                        prompt = get_prompt_for_row(selected_row)
                        
                        # Generation spinner
                        with st.spinner(f"Contacting OpenAI ({selected_model_name}). Please wait..."):
                            start_time = time.time()
                            content, prompt_tokens, comp_tokens, total_tokens, error = call_openai_model(
                                api_key=api_key,
                                model_id=model_id,
                                prompt=prompt,
                                temp=temperature,
                                top_p=top_p,
                                freq_penalty=frequency_penalty
                            )
                            elapsed_time = time.time() - start_time
                            
                        if error:
                            st.error(f"❌ Error calling OpenAI: {error}")
                        else:
                            # Clean response text (replace "a^1" with "Rs.")
                            if content:
                                content = content.replace("a^1", "Rs.")
                                
                            # Store results in session state
                            st.session_state["db_generated_content"] = content
                            st.session_state["db_prompt_tokens"] = prompt_tokens
                            st.session_state["db_completion_tokens"] = comp_tokens
                            st.session_state["db_total_tokens"] = total_tokens
                            st.session_state["db_elapsed_time"] = elapsed_time
                            st.session_state["db_generated_for_id"] = selected_id

                # Display results from session state if available and matching selected ID
                if "db_generated_content" in st.session_state and st.session_state.get("db_generated_for_id") == selected_id:
                    content = st.session_state["db_generated_content"]
                    prompt_tokens = st.session_state["db_prompt_tokens"]
                    comp_tokens = st.session_state["db_completion_tokens"]
                    total_tokens = st.session_state["db_total_tokens"]
                    elapsed_time = st.session_state["db_elapsed_time"]
                    
                    # Display Stats
                    st.success(f"Generation completed in {elapsed_time:.2f} seconds!")
                    
                    st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-card">
                                <div class="metric-value">{prompt_tokens}</div>
                                <div class="metric-label">Prompt Tokens</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{comp_tokens}</div>
                                <div class="metric-label">Completion Tokens</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{total_tokens}</div>
                                <div class="metric-label">Total Tokens</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{elapsed_time:.2f}s</div>
                                <div class="metric-label">Latency</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display generated text
                    st.markdown("### 📝 Generated Article")
                    st.markdown(content)
                    
                    # Clean title for file name
                    clean_title = "".join(c if c.isalnum() or c in (" ", "_", "-") else "" for c in selected_row['article_title'])
                    clean_title = clean_title.replace(" ", "_").lower()
                    file_name = f"{clean_title}_generated.md"
                    
                    # Action Buttons
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        st.download_button(
                            label="📥 Download Article as Markdown",
                            data=content,
                            file_name=file_name,
                            mime="text/markdown",
                            key="btn_db_download"
                        )
                    with btn_col2:
                        if st.button("🔄 New Article", key="btn_db_new_article"):
                            reset_db_state()
                            st.rerun()

    with tab2:
        st.markdown("### ✍️ Input Custom Brief & Rhetoric")
        st.markdown("All fields are compulsory except **Purpose/Objective** and **Additional Information**.")

        # Input fields sequential (empty by default)
        custom_title = st.text_input("Article Title *", value="", key="cust_title")
        custom_content_type = st.selectbox(
            "Type of Article *",
            options=['Article', 'How-to Guide', 'Listicle', 'Blog'],
            index=0,
            key="cust_content_type"
        )
        custom_word_count = st.text_input("Word Count *", value="", key="cust_word_count")
        custom_tone = st.selectbox(
            "Tone *",
            options=['Formal & Professional', 'Informal & Friendly', 'Technical'],
            index=0,
            key="cust_tone"
        )
        # Frozen Target Geography to 'India' (disabled)
        custom_geo = st.text_input("Target Geography *", value="India", disabled=True, key="cust_geo")
        custom_demography = st.text_input("Target Demography *", value="", key="cust_demography")
        custom_keywords = st.text_input("Primary Keywords *", value="", key="cust_keywords")
        custom_client_name = st.text_input("Client Name *", value="", key="cust_client_name")
        custom_client_desc = st.text_area("Client Description *", value="", key="cust_client_desc")
        # Frozen Client Industry to 'Finance' (disabled)
        custom_industry = st.text_input("Client Industry *", value="Finance", disabled=True, key="cust_industry")
        
        # Optional fields
        custom_purpose = st.text_area("Purpose/Objective (Optional)", value="", key="cust_purpose")
        custom_additional_info = st.text_area("Additional Information (Optional)", value="", key="cust_additional_info")
        
        custom_rhetoric = st.text_area("Rhetoric Structure *", value="", height=250, key="cust_rhetoric")

        # Generate Button
        if st.button("🚀 Generate Custom Article", key="btn_cust_generate"):
            if not api_key:
                st.error("API Key is missing. Please add `OPENAI_API_KEY` to your environment or `.env` file.")
            else:
                # Validate compulsory fields
                missing_fields = []
                if not custom_title.strip():
                    missing_fields.append("Article Title")
                if not custom_word_count.strip():
                    missing_fields.append("Word Count")
                if not custom_demography.strip():
                    missing_fields.append("Target Demography")
                if not custom_keywords.strip():
                    missing_fields.append("Primary Keywords")
                if not custom_client_name.strip():
                    missing_fields.append("Client Name")
                if not custom_client_desc.strip():
                    missing_fields.append("Client Description")
                if not custom_rhetoric.strip():
                    missing_fields.append("Rhetoric Structure")

                if missing_fields:
                    st.error(f"⚠️ Please fill in all compulsory fields: {', '.join(missing_fields)}")
                else:
                    # Treat empty optional fields as 'NA'
                    purpose_val = custom_purpose.strip() if custom_purpose.strip() else "NA"
                    additional_info_val = custom_additional_info.strip() if custom_additional_info.strip() else "NA"

                    # Build prompt
                    prompt = ARTICLE_GENERATION_PROMPT.format(
                        article_title=custom_title.strip(),
                        content_type=custom_content_type,
                        word_count=custom_word_count.strip(),
                        tone=custom_tone,
                        geographic_target=custom_geo,
                        demographic_target=custom_demography.strip(),
                        primary_keywords=custom_keywords.strip(),
                        client_name=custom_client_name.strip(),
                        client_description=custom_client_desc.strip(),
                        industry=custom_industry,
                        purpose_objective=purpose_val,
                        additional_information=additional_info_val,
                        rhetoric_structure=custom_rhetoric.strip()
                    )

                    with st.spinner(f"Contacting OpenAI ({selected_model_name}). Please wait..."):
                        start_time = time.time()
                        content, prompt_tokens, comp_tokens, total_tokens, error = call_openai_model(
                            api_key=api_key,
                            model_id=model_id,
                            prompt=prompt,
                            temp=temperature,
                            top_p=top_p,
                            freq_penalty=frequency_penalty
                        )
                        elapsed_time = time.time() - start_time

                    if error:
                        st.error(f"❌ Error calling OpenAI: {error}")
                    else:
                        if content:
                            content = content.replace("a^1", "Rs.")
                        
                        # Store in custom states
                        st.session_state["custom_generated_content"] = content
                        st.session_state["custom_prompt_tokens"] = prompt_tokens
                        st.session_state["custom_completion_tokens"] = comp_tokens
                        st.session_state["custom_total_tokens"] = total_tokens
                        st.session_state["custom_elapsed_time"] = elapsed_time

        # Display custom generation results if available
        if "custom_generated_content" in st.session_state:
            content = st.session_state["custom_generated_content"]
            prompt_tokens = st.session_state["custom_prompt_tokens"]
            comp_tokens = st.session_state["custom_completion_tokens"]
            total_tokens = st.session_state["custom_total_tokens"]
            elapsed_time = st.session_state["custom_elapsed_time"]

            st.success(f"Generation completed in {elapsed_time:.2f} seconds!")
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-value">{prompt_tokens}</div>
                        <div class="metric-label">Prompt Tokens</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{comp_tokens}</div>
                        <div class="metric-label">Completion Tokens</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{total_tokens}</div>
                        <div class="metric-label">Total Tokens</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{elapsed_time:.2f}s</div>
                        <div class="metric-label">Latency</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📝 Generated Article")
            st.markdown(content)

            # Clean custom title for download file name
            title_str = custom_title if custom_title.strip() else "custom_article"
            clean_title = "".join(c if c.isalnum() or c in (" ", "_", "-") else "" for c in title_str)
            clean_title = clean_title.replace(" ", "_").lower()
            file_name = f"{clean_title}_generated.md"

            # Action Buttons
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                st.download_button(
                    label="📥 Download Article as Markdown",
                    data=content,
                    file_name=file_name,
                    mime="text/markdown",
                    key="btn_cust_download"
                )
            with btn_col2:
                if st.button("🔄 New Article", key="btn_cust_new_article"):
                    reset_custom_state()
                    st.rerun()

if __name__ == '__main__':
    main()
