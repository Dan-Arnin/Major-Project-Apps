import streamlit as st
import base64
from data_processor import extract_data_from_pdf, extract_transcript, extract_text_from_url, split_into_chunks, create_pdf
from GraphRetrieval import GraphRAG
from llm_processor import llm_invoker
from datetime import time
import pickle, os

os.environ['OPENAI_API_KEY'] = st.secrets.OPENAI_API_KEY

def set_sidebar_state():
    st.session_state.sidebar_ = True

def main():
    st.cache(suppress_st_warning=True)
    st.set_page_config(page_title="AI Notes Maker", page_icon=":notebook_with_decorative_cover:")
    st.markdown("<style>body { background-color: black; color: white; }</style>", unsafe_allow_html=True)

    st.title("AI Notes Maker")
    notes_data = ""

    # Create placeholders for input fields
    content_type_placeholder = st.empty()
    link_placeholder = st.empty()
    file_upload_placeholder = st.empty()
    if "sidebar_" not in st.session_state:
        st.session_state.sidebar_ = False
    if "main_app" not in st.session_state:
        st.session_state.main_app = True
    if "grag" not in st.session_state:
        st.session_state.grag = GraphRAG()

    if st.session_state.main_app:

        with content_type_placeholder.container():
            content_type = st.selectbox("Select content type", ["PDF", "YouTube Link", "Website"])

            if content_type in ["YouTube Link", "Website"]:
                with link_placeholder.container():
                    link = st.text_input("Enter the link")
            else:
                with file_upload_placeholder.container():
                    file = st.file_uploader(f"Upload a {content_type.lower()} file")

        # Create a placeholder for the content
        content_placeholder = st.empty()
        submit_placeholder = st.empty()

        with submit_placeholder.container():
            if st.button("Submit"):
                content_type_placeholder.empty()
                link_placeholder.empty()
                file_upload_placeholder.empty()
                if content_type == "PDF":
                    if file is not None:
                        bytes_data = file.getvalue()
                        file2 = base64.b64encode(bytes_data)
                        notes_data = extract_data_from_pdf(file2)
                    else:
                        st.warning("Please upload a PDF file.")
                elif content_type == "YouTube Link":
                    if link:
                        notes_data = extract_transcript(link)
                    else:
                        st.warning("Please enter a YouTube link.")
                elif content_type == "Website":
                    if link:
                        st.markdown(f'<iframe src="{link}" width="800" height="600"></iframe>', unsafe_allow_html=True)
                        notes_data = extract_text_from_url(link)
                    else:
                        st.warning("Please enter a website link.")

        if notes_data != "":
            set_sidebar_state()
            submit_placeholder.empty()
            llm = llm_invoker()
            st.session_state.grag.constructGraph(notes_data)
            pre_notes_data = " ".join(st.session_state.grag.lines)
            pre_notes_list = split_into_chunks(pre_notes_data)

            summarized_list = []
            for i in pre_notes_list[:-1]:
                temp_data = llm.process_chunks(i)
                summarized_list.append(temp_data)

            summarized_final = ""
            for i in summarized_list:
                temp_data = llm.process_notes(i)
                summarized_final += temp_data

            pdf_data = create_pdf(summarized_final)
            

            # Replace the content placeholder with the summarized notes and download button
            with content_placeholder.container():
                st.write(summarized_final)
                if st.download_button(
                    label="Download PDF",
                    data=pdf_data,
                    file_name="workssss.pdf",
                    mime="application/pdf",
                    key = "download_button",
                ):
                    st.session_state.sidebar = True

    if st.session_state.sidebar_ == True:
        st.session_state.main_app = False
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