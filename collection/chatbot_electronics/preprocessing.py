import os
import ast

root_path = "/media/yanncauchepin/ExternalDisk/Datasets/NaturalLanguageProcessing/chatbot_electronics/"
input_json_path = "qa_Electronics.json"

def load_dataframe() :

    questions = list()
    answers = list()

    with open(os.path.join(root_path, input_json_path),'r') as input_file :
        for line in input_file :
            data = ast.literal_eval(line)
            questions.append(data['question'])
            answers.append(data['answer'])

    return {"questions" : questions, "answers" : answers}


if __name__ == '__main__' :

    """EXAMPLE"""

    df_electronics = load_dataframe()
    questions = df_electronics['questions']
    anwsers = df_electronics['answers']

    print(questions)
