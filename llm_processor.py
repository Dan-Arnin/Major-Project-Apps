from dotenv import load_dotenv
import os, json
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser



# OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
os.environ['OPENAI_API_KEY'] = st.secrets.OPENAI_API_KEY

class prompts():
    def __init__(self) -> None:
        pass

    def notes_maker_prompt(self):
        prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert notes compiler and summarizer"),
                ("user", """You will be provided with some data and you are to remove uneccesary data and summarize the data and make it look meaningful. Here is the data: 
                {input}
                 
                PROVIDE THE SUMMARIZED MEANINGFUL NOTES, ANYTHING ELSE IS NOT REQUIRED""")
            ])
        return prompt
    def process_recieved_notes(self):
        prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert teacher who corrects notes and data"),
                ("user", """You will be provided with a paragraph which lacks some context and might not seem continous. You are to correct that data by rearranging the sentences meaningfully, in the end the paragraph should be completely. Remove sentence which you feel are unwanted: 
                {input}
                 
                PROVIDE MEANINGFUL NOTES, ANYTHING ELSE IS NOT REQUIRED""")
            ])
        return prompt
    
    def virtual_teacher_(self):
        prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert teacher who teaches students with the provided material"),
                ("user", """You will be provided with some study material which you will use to teach your student. Provide data/notes in points and understandable way such as 
                 breaking down the information into small paragraphs.

                 Here is the information that has to be taught to the students:
                 {input}

                 Please only provide the information required nothing else. It should have a minimum of 200 words and a maximum of 300 words.
                 """)
            ])
        return prompt

    def test_question_generator(self):
        prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert question paper setter"),
                ("user", """You will be provided with some study material shall be the basis for the questions in the test. Provide 5 MCQs along with the options with one correct option
                 amongst the four, also provide a general descriptive question.

                 Provide the required data in the following JSON format:
                    <curly_brace>
                        "MCQ" : [
                            <curly_brace>
                                "question1" : "Question"
                                "options" : ["option1", "option2", "option3", "option4"]
                                "answer" : "option1"
                            </curly_brace>,
                            <curly_brace>
                                "question2" : "Question"
                                "options" : ["option1", "option2", "option3", "option4"]
                                "answer" : "option1"
                            </curly_brace>
                        ],
                        "descriptive_question" : [
                            "question" : "Question"
                        ]
                 
                    </curly_brace>
                 breaking down the information into small paragraphs.

                 Here is the data from which the questions have to be generated
                 {input}

                 Please only provide the JSON and nothing else.
                 """)
            ])
        return prompt
    
    def virtual_teacher_action_decider(self):
        prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert action decider who picks the right action as per the provided data"),
                ("user", """A teacher has instructed or taught something to her students. After going through the material provided by
                 the teacher the student either responds that he understood and the teacher can carry forward OR he can choose to 
                 ask a question that he wants to.
                 
                Here is the curriculum for that data
                {input}
                 
                JUST PROVIDE THE REQUIRED INTRODUCTIONS AND COURSE OUTLINE AS INSTRUCTED, NOTHING ELSE IS REQUIRED""")
            ])
        return prompt


class llm_invoker:
    def __init__(self) -> None:
        self.prompts = prompts()
        self.llm =  ChatOpenAI(api_key = os.getenv("OPEN_AI_KEY"))
        self.output_parser = StrOutputParser()
        self.teacher_index = 0
        self.chunks_3000 = []
    
    def process_chunks(self, chunk_data):
        chain = self.prompts.notes_maker_prompt() | self.llm | self.output_parser
        processed_data = chain.invoke({"input": chunk_data})
        return processed_data
    
    def process_notes(self, notes_data):
        chain = self.prompts.process_recieved_notes() | self.llm | self.output_parser
        data = chain.invoke({"input": notes_data})
        return data
    
    def process_teacher_data(self, teacher_data):
        chain = self.prompts.virtual_teacher_() | self.llm | self.output_parser
        data = chain.invoke({"input": teacher_data})
        return data
    
    def mcq_data(self, teacher_data):
        llm = self.llm
        llm.model_kwargs = {"response_format": { "type": "json_object" }}
        chain = self.prompts.test_question_generator() | llm | self.output_parser
        data = chain.invoke({"input": teacher_data})
        return json.loads(data)
    
    # def process_teacher_data(self, question_data):
    #     llm = self.llm
    #     llm.model_kwargs = {"response_format": { "type": "json_object" }}
    #     chain = self.prompts.virtual_teacher_() | self.llm | self.output_parser
    #     data = chain.invoke({"input": question_data})
    #     return data
