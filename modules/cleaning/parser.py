import re
import pandas as pd
from bs4 import BeautifulSoup

class Parser():
    
    def __init__(self):
        pass
    
    @staticmethod
    def html_beautiful_soup(html) :
        """BeautifulSoup module with the HTML Parser parameter is a HTML parser object.
        An example could be 'soup = BeautifulSoup(html_text, "html.parser")'."""
        return BeautifulSoup(html).get_text()


if __name__ == '__main__' :

    """EXAMPLE"""

    data = {
        'text': ["<html>This is a test review with <em>HTML</em> markup and some punctuation, like :) and :-)</html>",
                 "<html>:-) Another review with more <em>markup</em> and punctuation! </html>"]
        }
    df = pd.DataFrame(data)
    df_cleaned = Parser.imdb_df(df, 'text')
    print(f"- First test review :\n{df_cleaned.loc[0, 'text']}")
    print(f"- Second test review :\n{df_cleaned.loc[1, 'text']}")

    html = "<!DOCTYPE html><html><body><h1>My First Heading</h1><p>My first paragraph.</p></body></html>"
