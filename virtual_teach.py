import streamlit as st
from GraphRetrieval import GraphRAG
from data_processor import split_into_chunks
from data_processor import extract_data_from_pdf, extract_transcript, extract_text_from_url, split_into_chunks, create_pdf
from llm_processor import llm_invoker
import pickle, os
import base64

os.environ['OPENAI_API_KEY'] = st.secrets.OPENAI_API_KEY


def sidebar():
    # st.session_state.main_app = False
    with st.sidebar:
        st.title("Teacher: Clarify Your Doubts")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask me something"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            response = f"Teacher: {st.session_state.grag.queryLLM(prompt)}"
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


def teacher(content_type_placeholder):
    # print(st.session_state.tea)
    if st.session_state.counter < len(st.session_state.teacher_data):
        teach_data = f"Teacher: {st.session_state.teacher_data[st.session_state.counter]}"
        with st.chat_message("assistant"):
            st.markdown(teach_data)

        if st.button("Next"):
            st.session_state.counter += 1
            content_type_placeholder.empty()
    else:
        st.session_state.main_app = "TEST"
        st.session_state.sidebar_ = False
        print("kyaa")
        st.experimental_rerun()



def file_processor():
    # data = pickle.loads(file.read())

    # with open("data.pkl", "wb") as f:
    #     pickle.dump(data, f)

    # current_path = f"{os.getcwd()}/data.pkl"

    # st.session_state.grag.load_db(current_path)
    # print(st.session_state.grag.lines)
    st.session_state.sidebar_ = True
    pre_notes_data = " ".join(st.session_state.grag.lines)
    pre_notes_list = split_into_chunks(pre_notes_data)

    llm = llm_invoker()
    summarized_list = []
    for i in pre_notes_list[:-1]:
        temp_data = llm.process_chunks(i)
        summarized_list.append(temp_data)

    for i in summarized_list:
        temp_data = llm.process_teacher_data(i)
        st.session_state.teacher_data.append(temp_data)
    st.session_state.main_app = "TEACHER"

def test(data):
    print(data)
    for i, mcq in enumerate(data["MCQ"]):
        st.subheader(f"Question {i + 1}: {mcq['question{i+1}']}")
        st.session_state.user_answers[f"mcq_{i}"] = st.radio(f"Select your answer for Question {i + 1}", mcq['options'])

# Button to submit the answers
    if st.button("Submit"):
        st.subheader("Results")

        # Check MCQ answers
        correct_count = 0
        for i, mcq in enumerate(data["MCQ"]):
            if st.session_state.user_answers[f"mcq_{i}"] == mcq["answer"]:
                correct_count += 1
                st.write(f"Question {i + 1}: Correct")
            else:
                st.write(f"Question {i + 1}: Incorrect")

    st.write(f"Total Correct Answers: {correct_count} out of {len(data['MCQ'])}")
    pass

def main():
    st.set_page_config(
        page_title="Virtual Teacher", page_icon=":notebook_with_decorative_cover:"
    )
    st.markdown(
        "<style>body { background-color: black; color: white; }</style>",
        unsafe_allow_html=True,
    )

    st.title("Virtual Teacher")
    notes_data = ""

    if "main_app" not in st.session_state:
        st.session_state.main_app = "UPLOAD"
        st.session_state.sidebar_ = False
        st.session_state.grag = GraphRAG()
        st.session_state.teacher_data = []
        st.session_state.counter = 0
        st.session_state.user_answers = []

    content_type_placeholder = st.empty()
    file_upload_placeholder = st.empty()
    content_type_placeholder = st.empty()
    link_placeholder = st.empty()
    # test_placeholder = st.empty()

    if st.session_state.main_app == "UPLOAD":
            # file = st.file_uploader("Upload the Graph RAG data please")
        with content_type_placeholder.container():
            content_type = st.selectbox("Select content type", ["PDF", "YouTube Link", "Website"])

            if content_type in ["YouTube Link", "Website"]:
                with link_placeholder.container():
                    link = st.text_input("Enter the link")
            else:
                with file_upload_placeholder.container():
                    file = st.file_uploader(f"Upload a {content_type.lower()} file")
        content_type_placeholder = st.empty()
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
            submit_placeholder.empty()
            st.session_state.grag.constructGraph(notes_data)
            file_processor()

        # if file is not None:
        #     content_type_placeholder.empty()
        #     file_upload_placeholder.empty()
        #     file_processor(file)

    if st.session_state.sidebar_:
        sidebar()

    if st.session_state.main_app == "TEACHER":
        teacher_contents = st.empty()
        with content_type_placeholder.container():
            teacher(teacher_contents)

    # if st.session_state.main_app == "TEST":
    #     print("bartha idhe")
    #     content_type_placeholder.empty()
    #     llm = llm_invoker()
    #     teacher_data_string = " ".join(str(element) for element in st.session_state.teacher_data)
    #     mcq_data = llm.mcq_data(teacher_data_string)
    #     with content_type_placeholder.container():
    #         test(mcq_data)

if __name__ == "__main__":
    main()

