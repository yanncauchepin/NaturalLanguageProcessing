import os
import pandas as pd


class OpinionsPreprocessing():
    
    root_path = "/media/yanncauchepin/ExternalDisk/Datasets/NaturalLanguageProcessing/classifier_opinion/"
    input_path = {
        "amazon" : "sentiment_labelled_sentences/amazon_cells_labelled.txt",
        "imdb" : "sentiment_labelled_sentences/imdb_labelled.txt",
        "yeld" : "sentiment_labelled_sentences/yelp_labelled.txt",
        "comments" : "test.xls",
        "twitter_1" : "twitter_training.csv",
        "tiwtter_2" : "Mental-Health-Twitter.csv"
        }
    
    def __init__(self):
        pass
    
    def load_dataframe(dataset_name) :
    
        if dataset_name not in OpinionsPreprocessing.input_path.keys():
            raise ValueError("Dataset name not recognized."
                             f"Must be among {list(input_path.keys())}")
            
        dataset_path = os.path.join(OpinionsPreprocessing.root_path, OpinionsPreprocessing.input_path[dataset_name])
        if dataset_name == 'amazon':
            data = pd.read_csv(dataset_path, sep='\t', header=None)
            texts = data.iloc[:,0]
            opinions = data.iloc[:,-1]
        else:
            raise Exception('Dataset name recognized but not currently handled in the code.')
    
        return {"texts" : texts, "opinions" : opinions}


if __name__ == '__main__' :

    """EXAMPLE"""

    df_amazon = OpinionsPreprocessing.load_dataframe("amazon")
    texts = df_amazon['texts']
    opinions = df_amazon['opinions']

    print(texts)
    print(opinions)
