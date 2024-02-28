import os
import pandas as pd

root_path = "/media/yanncauchepin/ExternalDisk/Datasets/NaturalLanguageProcessing/classifier_questions/"
input_path = "labelled_questions.txt"
    

def load_dataframe() :
    
    dataset_path = os.path.join(root_path, input_path)
    data = open(dataset_path, 'r+')
    dataframe = pd.DataFrame(data.readlines(), columns=['Question'])
    dataframe['QType'] = dataframe.Question.apply(lambda x: x.split(' ',1)[0])
    dataframe['Question'] = dataframe.Question.apply(lambda x: x.split(' ',1)[1])
    dataframe["QType-Coarse"] = dataframe.QType.apply(lambda x: x.split(':')[0])
    dataframe["QType-Fine"] = dataframe.QType.apply(lambda x: x.split(':')[1])

    return dataframe


if __name__ == '__main__' :

    """EXAMPLE"""

    df_questions = load_dataframe()

    print(df_questions)
