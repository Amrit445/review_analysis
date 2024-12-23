import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup

url= "https://www.flipkart.com/l-oral-paris-extraordinary-oil-shampoo-340-ml-conditioner-180-serum-30/p/itm0a099856ea57f?pid=CBKGZ5EZ6EHWJ23H&ssid=pntzsmsrps0000001734624826105"

r=requests.get(url)
st.
