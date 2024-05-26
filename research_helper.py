import streamlit as st
from research_paper import arxiv_papers
from GraphRetrieval import GraphRAG
import os

os.environ['OPENAI_API_KEY'] = st.secrets.OPENAI_API_KEY

if "main_page" not in st.session_state:
    st.session_state.main_page = "initial"
    st.session_state.grag = GraphRAG()

def main():
    content_placeholder = st.empty()

    if st.session_state.main_page == "initial":
        with content_placeholder.container():
            st.title("Research Paper Helper:")
            query = st.text_input("Enter something to get papers")

            if st.button("submit"):
                if "paper_data" not in st.session_state:
                    st.session_state.paper_data = arxiv_papers(query)
                    st.session_state.main_page = "paper_list"

    if st.session_state.main_page == "paper_list":
        content_placeholder.empty()
        with content_placeholder.container():
            if st.session_state.paper_data:
                st.header("_Papers found:_", divider="rainbow")
                for index, paper in enumerate(st.session_state.paper_data):
                    st.write(f"{index + 1}. {paper['title']} - {paper['authors']}")
                
                select_paper = st.text_input("Please enter the number of the paper you want to study")
                print(select_paper)
                if st.button("Lets Chat"):
                    st.session_state.grag.constructGraph(st.session_state.paper_data[int(select_paper)]["content"][:10000])
                    st.session_state.main_page = "chatbot"
    
    if st.session_state.main_page == "chatbot":
        content_placeholder.empty()
        st.title("Ask me anything from the document")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask me something"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            response = f"Bot: {st.session_state.grag.queryLLM(prompt)}"
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})




if __name__ == "__main__":
    main()