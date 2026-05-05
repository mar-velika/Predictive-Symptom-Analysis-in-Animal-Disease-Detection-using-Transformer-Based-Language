import streamlit as st

st.title("Simple Streamlit App")
st.header("Welcome")
st.write("This is a basic streamlit app.")

name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}! \nHow are you doing today?")

number = st.slider("Pick your age:", 0, 100)
st.write(f"You are {number} years old.")