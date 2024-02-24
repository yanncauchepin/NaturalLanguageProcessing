import requests


def html_website(url) :
    """To test"""
    return request.get(url).text


if __name__ == '__main__' :

    """EXAMPLE"""

    url = "https://webscraper.io/test-sites/e-commerce/allinone/computers/laptops"
