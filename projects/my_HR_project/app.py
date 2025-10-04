import streamlit as st
from streamlit_chat import message
from model import get_reply


def on_input_change():
    prompt = st.session_state.user_input

    if st.session_state['exchange'] is not None:
        st.session_state.user_input = ""
        return

    st.session_state.user_input = ""

    # Use the get_reply function to get GPT2 reply.
    try:
        reply = get_reply(prompt)
        # Ensure UI shows only the completion, not the echoed prompt
        if isinstance(reply, str) and isinstance(prompt, str) and reply.startswith(prompt):
            reply = reply[len(prompt):].lstrip()
    except Exception as e:
        # Surface the model error directly so the user sees why generation failed.
        reply = f"Error in get_reply: {e}"

    st.session_state['exchange'] = {'user': prompt, 'reply': reply}

def on_btn_click():
    st.session_state['exchange'] = None

# Initialize session state only if it doesn't already exist
if 'exchange' not in st.session_state:
    st.session_state['exchange'] = None

st.title("Hi, write your job description and a list of requirements like in the example below:")

st.write(
    "HR recommendation systems must continuously adapt to evolving candidate profiles and job postings."
    " Static, batch-trained models struggle to stay up-to-date and often fail to maintain relevance over time without costly retraining."
    " This PhD project explores how continual learning (CL) and online learning to rank (OLTR) can enable adaptive,"
    " scalable HR systems built on structured JSON data—including parsed resumes and job descriptions."
    " Key research questions include: Incremental adaptation, Real-time ranking, Domain-specific embeddings,"
    " Conducted in partnership with HrFlow.ai, this research is anchored in real-world, structured HR datasets and production systems."
    " Expected contributions include new algorithms for lifelong ranking and dynamic representation learning tailored to HR applications.\n\n"
    "List of requirements: automnomy, creativity, strong coding skills in python, machine learning, deep learning, NLP, "
)

chat_placeholder = st.empty()

with chat_placeholder.container():
    if st.session_state['exchange'] is not None:
        message(st.session_state['exchange']['user'], is_user=True, key="user")
        message(st.session_state['exchange']['reply'], key="bot")

    st.button("Retry", on_click=on_btn_click)


if st.session_state['exchange'] is not None:
    pass
else:
    with st.container():
        st.text_input("User Input:", on_change=on_input_change, key="user_input")
