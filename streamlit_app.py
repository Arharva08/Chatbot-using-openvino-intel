import streamlit as st
import requests

st.set_page_config(page_title="Medical & Drug Assistant Bot", page_icon=":pill:")

st.title("Medical & Drug Assistant ChatBot")

st.write("Ask any medical or drug-related question and get helpful responses.")

question = st.text_input("Your question here:")

if st.button("Ask"):
    if question:
        with st.spinner("Getting response..."):
            response = requests.post("http://localhost:5000/ask", json={"question": question})
            if response.status_code == 200:
                st.success(response.json()['response'])
            else:
                st.error("Error: Could not get response from the server.")
    else:
        st.warning("Please enter a question.")

st.sidebar.title("About")
st.sidebar.info(
    """
    This is a medical and drug assistant bot built using OpenVINO and a language model.
    """
)
