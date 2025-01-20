
import streamlit as st

state = st.session_state
def search() -> None:
    if 'go_clicked' not in state:
        state.go_clicked = False

    def click_button():
        state.go_clicked = True

    search_query = st.text_input(label="Search", value="red carpet")
    st.button(label="Go", on_click=click_button)

    if state.go_clicked:
        state.search_results = state.vectorstore.similarity_search(search_query)
        st.write(state.search_results)
        state.go_clicked = False

search()