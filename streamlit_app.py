import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page

# Sidebar selection should come BEFORE showing any page
page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))

# Show the selected page
if page == "Predict":
    show_predict_page()
else:
    show_explore_page()
