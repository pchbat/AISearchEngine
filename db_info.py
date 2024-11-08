from myfunctions import  show_files_in_database
import streamlit as st

def db_info():
    st.title("Database Info")
    show_files_in_database(st.session_state['vectorstore'])