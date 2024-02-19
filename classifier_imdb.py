import collection.classifier_imdb.preprocessing as imdb_preprocessing

import modules.cleaning.parser as cleaning_parser
import modules.cleaning.tokenizer as cleaning_tokenizer
import modules.cleaning.stemming as cleaning_stemming

if __name__ == '__main__' :

    df_imdb = imdb_preprocessing.load_dataframe()
    df_imdb= cleaning_parser.imdb_df_parser(df_imdb, 'review')

    imdb_tokens = cleaning_tokenizer.tokenizer(df_imdb.loc[0, 'review'])
    imdb_cleaned_tokens = cleaning_stemming.porter_stemming(imdb_tokens)
    print(cleaning_stemming.stop_word(imdb_cleaned_tokens))
