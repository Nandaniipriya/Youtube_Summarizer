import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders.url import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import re
import urllib.parse

## streamlit APP
st.set_page_config(page_title="Langchain: Summarize Text From YT or Website", page_icon="▶️")
st.title("▶️ LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Get the Groq API Key and url field to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")
generic_url = st.text_input("URL", label_visibility="collapsed")

def extract_youtube_id(url):
    """Extract YouTube video ID from URL"""
    # Clean up the URL first
    url = url.strip()
    
    # Handle youtu.be links
    if 'youtu.be' in url:
        parts = url.split('/')
        for part in parts:
            if len(part) == 11 and '?' not in part:
                return part
            elif len(part) > 11 and '?' in part:
                return part.split('?')[0]
    
    # Handle youtube.com links
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:shorts\/)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # Parse URL parameters as last resort
    try:
        parsed_url = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        if 'v' in query_params:
            return query_params['v'][0]
    except:
        pass
    
    return None

def get_youtube_transcript(video_id):
    """Get transcript using youtube_transcript_api directly with multiple fallbacks"""
    try:
        # Import here to avoid issues if not installed
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

        # Try with multiple language options
        language_options = [
            ['en'],
            ['en-US'],
            ['en-GB'],
            ['a.en'],  # Auto-generated English
            []  # Any available language
        ]
        
        transcript_text = None
        last_error = None
        
        for languages in language_options:
            try:
                if languages:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
                else:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                    
                transcript_text = " ".join([d['text'] for d in transcript_list])
                if transcript_text:
                    return transcript_text, None
            except Exception as e:
                last_error = str(e)
                continue
        
        # If we got here, all direct methods failed, try translation method
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to find any transcript and translate to English if needed
            found_transcript = None
            
            # Try English first
            try:
                found_transcript = transcript_list.find_transcript(['en'])
            except:
                # Get the first available transcript
                for transcript in transcript_list:
                    found_transcript = transcript
                    break
            
            if found_transcript:
                # Translate to English if it's not already in English
                if found_transcript.language_code != 'en':
                    found_transcript = found_transcript.translate('en')
                
                result = found_transcript.fetch()
                transcript_text = " ".join([d['text'] for d in result])
                if transcript_text:
                    return transcript_text, None
        except Exception as e:
            last_error = f"{last_error} | Translation method failed: {str(e)}"
        
        return None, f"All transcript methods failed. Last error: {last_error}"
        
    except ImportError:
        return None, "youtube_transcript_api not installed. Install with: pip install youtube_transcript_api"

if st.button("Summarize the content from Youtube or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video url or website url")
    else:
        try:
            with st.spinner("Loading content..."):
                ## Initialize the LLM
                llm = ChatGroq(model="qwen-2.5-32b", groq_api_key=groq_api_key)
                
                ## Set up the prompt template
                prompt_template = """
                Provide a comprehensive summary of the following content in 400 words.
                Focus on the main ideas, key points, and important details.
                
                Content: {text}
                
                Summary:
                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                
                ## loading the website or yt video data
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    st.info("Processing YouTube video...")
                    youtube_id = extract_youtube_id(generic_url)
                    
                    if not youtube_id:
                        st.error("Couldn't extract YouTube video ID from the URL")
                        st.stop()
                        
                    st.write(f"Extracted video ID: {youtube_id}")
                    
                    # Skip all libraries that might cause HTTP 400 errors and go directly to the transcript API
                    transcript_text, error = get_youtube_transcript(youtube_id)
                    
                    if transcript_text:
                        st.success("Successfully extracted transcript")
                        docs = [Document(
                            page_content=transcript_text,
                            metadata={"source": f"https://www.youtube.com/watch?v={youtube_id}", "title": f"YouTube Video {youtube_id}"}
                        )]
                    else:
                        st.error(f"Failed to extract transcript: {error}")
                        st.stop()
                else:
                    st.info("Processing website content...")
                    try:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
                            }
                        )
                        docs = loader.load()
                    except Exception as url_error:
                        st.error(f"Error loading website content: {url_error}")
                        st.stop()
                
                # Check if we have content to summarize
                if not docs or not docs[0].page_content:
                    st.error("No content could be extracted from the provided URL.")
                    st.stop()
                
                # Show content preview
                content_preview = docs[0].page_content[:500] + "..." if len(docs[0].page_content) > 500 else docs[0].page_content
                with st.expander("Content Preview"):
                    st.text(content_preview)
                
                # Check content length and split if necessary
                content_length = sum(len(doc.page_content) for doc in docs)
                
                with st.spinner("Summarizing content..."):
                    if content_length > 25000:
                        st.info("Content is large, splitting into chunks for processing...")
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=10000,
                            chunk_overlap=1000
                        )
                        split_docs = text_splitter.split_documents(docs)
                        chain = load_summarize_chain(
                            llm, 
                            chain_type="map_reduce",
                            map_prompt=prompt,
                            combine_prompt=prompt,
                            verbose=True
                        )
                        output_summary = chain.run(split_docs)
                    else:
                        chain = load_summarize_chain(
                            llm, 
                            chain_type="stuff", 
                            prompt=prompt,
                            verbose=True
                        )
                        output_summary = chain.run(docs)
                
                st.subheader("Summary")
                st.success(output_summary)
                
                # Display content length information
                st.info(f"Content length: {content_length} characters")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)