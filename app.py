import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructureURLLoader
from langchain_community.document_loaders.url import UnstructuredURLLoader

## streamlit APP
st.set_page_config(page_title="Langchain:Summarize Text From YT or Website",page_icon="▶️")
st.title("▶️ LangChain:Summarize Text From YT or Website")
st.subheader('Summarize URL')

##Get the Groq API Keyand url field to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="",type="password")
generic_url=st.text_input("URL",label_visibility="collapsed")
    
if st.button("Summarize the content from Youtube or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid UrL.It can may be a YT video utl or website url")
    else:
         try:
             with st.spinner("Waiting..."):
                 ## loading the website or yt video data
                if "youtube.com" in generic_url:
                     loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"})