import numpy as np

import samples.chatbot_software.preprocessing as software_preprocessing
import samples.chatbot_electronics.preprocessing as electronics_preprocessing

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer


class ChatBot():
    
    def __init__(self):
        self.questions = list()
        self.answers = list()
        self.init = False
        self.update = False
    
    @staticmethod
    def check_questions_answers(questions, answers):
        if isinstance(questions, str) and isinstance(answers, str):
            questions = [questions]
            answers = [answers]
        if isinstance(questions, list) and isinstance(answers, list):
            '''
            if len(questions) != len(answers):
                raise Exception(f'Length of questions {len(questions)} must be '
                                f'commensurate to the length of anwsers {len(answers)}.')
            '''
            pass
        else:
            raise Exception('Questions and answers must be provided either in list '
                            'or by string for single question and its answer.')
        return questions, answers
    
    @staticmethod
    def check_question(question):
        if isinstance(question, str):
            question = [question]
        # other treatment
        return question
   
    def append_questions_answers(self, questions, answers):
        questions, anwsers = ChatBot.check_questions_answers(questions, answers)
        self.questions.extend(questions)
        self.answers.extend(answers)
        self.update = False
             
    def __init_chatbot(self):
        # Check dataset excited
        self.vectorizer = CountVectorizer(stop_words='english')
        self.X_vec = self.vectorizer.fit_transform(self.questions)
        self.tfidf_transformer = TfidfTransformer(norm='l2')
        self.X_tfidf = self.tfidf_transformer.fit_transform(self.X_vec)
        self.init = True
        self.update = True
         
    def answer(self, question):
        # Check self.init = True
        # Check self.update = True
        question = ChatBot.check_question(question)
        Y_vec = self.vectorizer.transform(question)
        Y_tfidf = self.tfidf_transformer .fit_transform(Y_vec)
        angle = np.rad2deg(np.arccos(max(cosine_similarity(Y_tfidf, \
                           self.X_tfidf)[0])))
        if angle > 60 :
            return "Sorry, I do not understand that question."
        else:
            return self.answers[np.argmax(cosine_similarity(Y_tfidf, self.X_tfidf)[0])]
    
    def run(self):
        self.__init_chatbot()
        user = input("Please enter your username: ")
        print("Q&A support: Hi, welcome to Q&A support. How can I help you?")
        print("(Insert 'Bye' to end the chatbot.)")
        while True:
            question = input(f"{user}: ")
            if question.lower() == 'bye':
                print("Q&A support: Bye!")
                break
            else:
                print(f"Q&A support: {self.answer(question)}")
        
   
class SoftwareChatBot():
    
    def __init__(self):
        pass
    
    @staticmethod
    def run():
        chatbot = ChatBot()
        df_software = software_preprocessing.load_dataframe()
        questions = df_software['questions']
        answers = df_software['answers']
        chatbot.append_questions_answers(questions, answers)
        chatbot.run()
        
       
class ElectronicsChatBot():
    
    def __init__(self):
        pass
    
    def run():
        chatbot = ChatBot()
        df_electronics = electronics_preprocessing.load_dataframe()
        questions = df_electronics['questions']
        anwsers = df_electronics['answers']
        chatbot.append_questions_answers(questions, answers)
        chatbot.run()

    
if __name__ == '__main__':
    #SoftwareChatBot.run()
    #ElectronicsChatBot.run()
    