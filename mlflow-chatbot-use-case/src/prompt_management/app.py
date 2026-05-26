import json

import streamlit as st
from prompt_registry import PromptManagement, RegisterPrompt

st.title("Prompt Management")

pm = PromptManagement()

# -----------------------
# EXPERIMENT (AUTO SWITCH)
# -----------------------
st.subheader("Experiment")

if "experiment" not in st.session_state:
    st.session_state.experiment = "chatbot-prompts"

experiment_name = st.text_input(
    "MLflow Experiment Name", value=st.session_state.experiment
)

if experiment_name != st.session_state.experiment:
    st.session_state.experiment = experiment_name
    pm.set_experiment(experiment_name)
    st.info(f"Switched to experiment: {experiment_name}")

pm.set_experiment(st.session_state.experiment)

# -----------------------
# REGISTER PROMPT
# -----------------------
st.subheader("Register Prompt")

name = st.text_input("Prompt Name")

template_type = st.radio("Template Type", ["Text", "Chat (JSON)"])

if template_type == "Text":
    template = st.text_area(
        "Template (use {{variables}})", value="Summarize the following text: {{text}}"
    )
else:
    template_json = st.text_area(
        "Chat Template JSON",
        value=json.dumps(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "{{question}}"},
            ],
            indent=2,
        ),
    )

# -----------------------
# OPTIONAL SETTINGS
# -----------------------
with st.expander("Optional Settings"):
    commit_message = st.text_input(
        "Commit Message", placeholder="e.g. improve summarization prompt"
    )

    tags_input = st.text_area("Tags (key:value, comma separated)")

    st.caption("Example:")
    st.code(
        "author:author@example.com, task:summarization, language:en", language="text"
    )

    # -----------------------
    # RESPONSE FORMAT
    # -----------------------
    st.markdown("### Response Format (optional)")

    response_format_input = st.text_area("Response Format JSON")

    st.caption("Example:")

    st.code(
        json.dumps(
            {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                    "word_count": {"type": "integer"},
                },
            },
            indent=2,
        ),
        language="json",
    )

    # -----------------------
    # MODEL CONFIG
    # -----------------------
    st.markdown("### Model Config")

    provider = st.text_input("Provider (openai / anthropic / google)")
    model_name = st.text_input("Model Name (required for config)")

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input("Temperature", value=0.7)
        top_p = st.number_input("Top P", value=1.0)
        frequency_penalty = st.number_input("Frequency Penalty", value=0.0)
    with col2:
        max_tokens = st.number_input("Max Tokens", value=500)
        top_k = st.number_input("Top K", value=0)
        presence_penalty = st.number_input("Presence Penalty", value=0.0)

    stop_sequences_input = st.text_input(
        "Stop Sequences (comma separated)", placeholder="END,STOP"
    )

    # -----------------------
    # EXTRA PARAMS
    # -----------------------
    st.markdown("### Extra Params (Provider-specific)")

    extra_params_input = st.text_area("Extra Params JSON")

    st.caption("Example:")

    st.code(
        json.dumps(
            {
                "seed": 42,
                "user": "user-123",
                "service_tier": "default",
                "logit_bias": {"50256": -100},
            },
            indent=2,
        ),
        language="json",
    )


# -----------------------
# HELPERS
# -----------------------
def parse_tags(tags_str):
    if not tags_str:
        return None
    return dict(item.split(":", 1) for item in tags_str.split(",") if ":" in item)


def parse_list(input_str):
    if not input_str:
        return None
    return [x.strip() for x in input_str.split(",") if x.strip()]


def parse_json_safe(text):
    if not text or text.strip() == "":
        return {}
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


# -----------------------
# REGISTER ACTION
# -----------------------
if st.button("Register Prompt"):
    try:
        final_template = (
            template if template_type == "Text" else json.loads(template_json)
        )

        model_settings = None
        if model_name:
            from mlflow.entities.model_registry.prompt_version import PromptModelConfig

            model_settings = PromptModelConfig(
                provider=provider or None,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k if top_k != 0 else None,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop_sequences=parse_list(stop_sequences_input) or [],
                extra_params=parse_json_safe(extra_params_input) or {},
            )

        params = RegisterPrompt(
            name=name,
            template=final_template,
            commit_message=commit_message or None,
            tags=parse_tags(tags_input),
            response_format=parse_json_safe(response_format_input),
            model_settings=model_settings,
        )

        result = pm.register_prompt(params)
        st.success(result)

    except Exception as e:
        st.error(f"Error: {e}")


# streamlit run app.py dev