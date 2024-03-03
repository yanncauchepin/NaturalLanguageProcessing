import os
import json
import pandas as pd

root_path = "/media/yanncauchepin/ExternalDisk/Datasets/NaturalLanguageProcessing/classifier_sarcasm/"
input_path = "sarcasm/Sarcasm_Headlines_Dataset_v2.json"

def load_dataframe() :
    
    def parse_data(file):
        for l in open(file,'r'):
            yield json.loads(l)
    
    dataset_path = os.path.join(root_path, input_path)
    data = list(parse_data(dataset_path))
    dataframe = pd.DataFrame(data)
    dataframe.pop('article_link')

    return dataframe


if __name__ == '__main__' :

    df_sarcasm = load_dataframe()
    print(df_sarcasm)
