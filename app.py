import os
import time
import gradio as gr
from dotenv import load_dotenv
from pathlib import Path
import re
import json
from langchain_community.document_loaders import JSONLoader
# Import Document from your LangChain module. 
# (Adjust the import if your version of LangChain uses a different path.)
from langchain_core.documents import Document  

# Import additional libraries from LangChain
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables for Hugging Face and OpenAI
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# -------------------------------
# Utility Functions
# -------------------------------

def flatten_metadata(metadata):
    """Helper function to flatten dictionary metadata into a string."""
    if isinstance(metadata, dict):
        return " | ".join([f"{key}: {value}" for key, value in metadata.items()])
    return str(metadata)  # If it's not a dict, just return the string version

def metadata_func(record, additional_fields=None):
    is_winner = record.get("Ranking", "").lower() == "winner"
    
    return {
        "Project Title": record.get("Title", ""),
        "Organization": record.get("Organization", ""),
        "LA 2050 Grant Status": record.get("Ranking", ""),
        "Impact Metrics": record.get("Impact Metrics", ""),
        "LA 2050 Year": record.get("Year", ""),
        "Organizations urls": flatten_metadata({
            "Organization website": record.get("Website", ""),
            "Organization newsletter": record.get("Newsletter", ""),
            "volunteer": record.get("Volunteer", ""),
            "LA2050 website": record.get("LA2050", "")
        }),
        "social": flatten_metadata({
            "twitter": record.get("Twitter", ""),
            "instagram": record.get("Instagram", ""),
            "facebook": record.get("FaceBook", "")
        }),
        "working_area": record.get("Working Areas in LA", ""),
        "zipcode": record.get("Zipcode", "")
    }
# Load the JSON data with custom metadata and content key
loader = JSONLoader(
    file_path='data.json',
    jq_schema='.[]',
    content_key='Summary',
    metadata_func=metadata_func
)

data = loader.load()


# Use a text splitter to create chunks from the documents.
# (If you find that key fields are getting split, consider implementing a custom splitter.)
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1600,  
    chunk_overlap=150,  
    add_start_index=True,
    separators=["\n\n", "\n", ". ", " ", ""] 
)
def split_document_with_metadata(document):
    # Split the document text into chunks.
    chunks = text_splitter.split_text(document.page_content)
    # Ensure every chunk has the complete original metadata.
    return [Document(page_content=chunk, metadata=document.metadata) for chunk in chunks]

all_splits = []
for doc in data:
    all_splits.extend(split_document_with_metadata(doc))
# -------------------------------
# Set Up Retrievers
# -------------------------------

# Create a Chroma vector store using the document splits.
persist_directory = "path_to_persist_directory"

# Check if the directory exists and contains a persisted vector store
if os.path.exists(persist_directory):
    # Attempt to load the existing vector store
    try:
        vectorstore = Chroma.load(persist_directory, embedding_function=OpenAIEmbeddings())
        print("Loaded existing vector store from persisted directory.")
    except Exception as e:
        print(f"Error loading vector store: {e}. Proceeding to create a new one.")
        # Fallback to creating a new vector store if loading fails
        vectorstore = Chroma.from_documents(
            documents=all_splits, 
            embedding=OpenAIEmbeddings(), 
            persist_directory=persist_directory
        )
        print("Created new vector store and persisted embeddings.")
else:
    # Create a new vector store if the directory doesn't exist
    vectorstore = Chroma.from_documents(
        documents=all_splits, 
        embedding=OpenAIEmbeddings(), 
        persist_directory=persist_directory
    )
    print("Created new vector store and persisted embeddings.")

# Create a BM25 retriever from the document splits.
bm25_retriever = BM25Retriever.from_documents(all_splits)
ensemble_retriever = EnsembleRetriever(
    retrievers=[
        vectorstore.as_retriever(),
        bm25_retriever
    ],
    weights=[0.9, 0.1]
)
retriever = ensemble_retriever

# -------------------------------
# Prepare Retrieval and Generation Chain
# -------------------------------

system_prompt = (
    "You are the LA2050 Navigator, an AI-powered chatbot created to help users discover organizations and community initiatives featured in the Goldhirsh Foundation’s LA2050 Ideas Hub. "
    "Your role is to deliver succinct, personalized recommendations, guide users toward supporting these initiatives, and answer questions about the Goldhirsh Foundation, LA2050, and its projects. "
    "When responding, include the full name of the organization, a brief (1-2 sentence) description, and a link to its website (labeled as Organization website) or social media; (please do not alter the URL). "
    "If an organization’s personal website is unavailable, refer to its LA2050 URL. "
    "Prioritize nonprofit organizations designated as 'winners' by the Goldhirsh Foundation and those with multiple proposal submissions. "
    "If a user inquires about the LA2050 grant winners for a specific year, be sure to look out for 'LA 2050 Grant Status'-explicitly noting if the organization was awarded the grant that year(disregard if it has 'Submitted)'. "
    "Use the data files as your primary source of information. These files have been pre-processed into context-rich segments using a recursive text-splitting approach to ensure key details are preserved. "
    "If some information is missing, acknowledge it and direct the user to additional resources. "
    "Maintain a polite, helpful, respectful, and enthusiastic tone at all times. "
    "If the user responds with a follow-up confirmation (e.g., 'yes') after an initial answer, please expand on that topic with further details. "
    "\n\nIMPORTANT: Answer the question using ONLY the information provided in the following documents. DO NOT invent or include any organizations that are not present in the retrieved evidence. "
    "Before giving your final answer, perform the following steps: "
    "Step 1: Identify all organizations mentioned in the retrieved documents. "
    "Step 2: Check if there are any organizations beyond those provided that could be considered 'new'. "
    "Step 3: If no additional organizations exist, clearly state that based on the current dataset, these are all the organizations we have information on. "
    "\n\n{context}"
)




prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Build the chain that will combine documents with the prompt.
question_answer_chain = create_stuff_documents_chain(ChatOpenAI(model_name="gpt-4o-mini", temperature=0), prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def post_process_answer(answer, retrieved_docs):
    """
    Append a disclaimer to the answer confirming that only organizations from the retrieved documents were used.
    (A more advanced implementation might parse and filter out any hallucinated names.)
    """
    # Extract allowed organization names from retrieved docs.
    allowed_orgs = {doc.metadata.get("Organization", "").strip() for doc in retrieved_docs if doc.metadata.get("Organization", "").strip()}
    disclaimer = "\n\n[Answer verified against retrieved documents: Only organizations present in the evidence were included. Allowed organizations: " + ", ".join(sorted(allowed_orgs)) + ".]"
    return answer + disclaimer

def debug_retrieved_docs(user_input):
    retrieved_docs = retriever.get_relevant_documents(user_input)
    print(f"DEBUG: Retrieved {len(retrieved_docs)} documents.")
    for i, doc in enumerate(retrieved_docs):
        print(f"Doc {i+1}: {doc.metadata}")
    return retrieved_docs
# -------------------------------
# Gradio Interface and Conversation Handling
# -------------------------------

green_theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#00A168",
        c100="#57B485",
        c200="#D7ECE0",
        c300="#FFFFFF",
        c400="#EAE9E9",
        c500="#000000",
        c600="#3A905E",
        c700="#2A774A",
        c800="#1A5E36",
        c900="#0A4512",
        c950="#052A08"
    ),
    font=[gr.themes.GoogleFont('Space Grotesk'), 'ui-sans-serif', 'system-ui', 'sans-serif']
).set(
    body_background_fill='#00A168',
    body_text_color='#000000',
    background_fill_primary='#FFFFFF',
    background_fill_secondary='#FFFFFF',
    border_color_accent='#57B485',
    border_color_accent_subdued='#EAE9E9',
    color_accent='#57B485',
    color_accent_soft='#D7ECE0',
    checkbox_background_color='#FFFFFF',
    button_primary_background_fill='#57B485',
    button_primary_background_fill_hover='#3A905E',
    button_secondary_background_fill='#D7ECE0',
    button_secondary_text_color='#000000'
)


def message_and_history(message, history):
    # Initialize conversation with a welcome message if history is empty.
    if not history:
        history = [{"role": "assistant", "content": "<b>LA2050 Navigator:</b><br> Welcome to the LA2050 ideas hub! How can I help you today?"}]
    
    # Handle if message is provided as a string or a dict.
    user_text = message if isinstance(message, str) else message.get("text", "")
    history.append({"role": "user", "content": user_text})
    
    time.sleep(1)

    if not user_text:
        history.append({"role": "assistant", "content": "<b>LA2050 Navigator:</b><br> Please enter a valid message."})
        yield history, history
        return

    # Combine the most recent conversation turns, excluding the assistant's prefix.
    conversation_context = "\n".join(
        [f"{msg['role']}: {msg['content'].replace('<b>LA2050 Navigator:</b><br>', '')}" for msg in history[-1:]]
    )
 
    retrieved_docs = retriever.invoke(conversation_context)

    print(f"DEBUG: Retrieved {len(retrieved_docs)} documents.")
    for i, doc in enumerate(retrieved_docs):
        # Print out key metadata fields to verify correctness.
        print(f"Doc {i+1} Page Content: {doc.page_content}")
    chain_input = {"input": conversation_context}
    
    try:
        response = rag_chain.invoke(chain_input)
        answer = response["answer"]
        # Post-process the answer to append a disclaimer verifying the evidence.
       
    except Exception as e:
        answer = f"An error occurred: {e}"

    # Remove the prefix if the model includes it.
    if answer.startswith("<b>LA2050 Navigator:</b><br>"):
        answer = answer[len("<b>LA2050 Navigator:</b><br>"):]
    
    # Initialize the assistant's response with the prefix.
    assistant_response = {"role": "assistant", "content": "<b>LA2050 Navigator:</b><br> "}
    history.append(assistant_response)

    # Stream the answer character by character.
    for character in answer:
        assistant_response["content"] += character
        yield history, history

    # Finalize the answer.
    history[-1]["content"] = assistant_response["content"]
    yield history, history

# Set Gradio to light mode via JavaScript
js_func = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

css = """
.chat-header {
    text-color: #FFFFFF;
    text-align: center;
}
.gradio-container .prose .chat-header h1 {
    color: #FFFFFF;
    text-align: center;
}
"""

with gr.Blocks(theme=green_theme, js=js_func, css=css) as block:
    gr.HTML('<div class="chat-header"><h1>LA2050 Navigator</h1></div>')
    
    chatbot = gr.Chatbot(
        value=[{"role": "assistant", "content": "<b>LA2050 Navigator:</b><br> Welcome to the LA2050 ideas hub! How can I help you today?"}],
        type="messages",
        bubble_full_width=False
    )
    
    state = gr.State([])
    
    message = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Type a message",
        label="",
        elem_classes="custom-textbox",
        scale=3,
        show_label=False
    )
    
    # When a message is submitted, the function sends the recent conversation history along with the new input.
    message.submit(
        message_and_history,
        inputs=[message, state],
        outputs=[chatbot, state]
    ).then(
        lambda: "", inputs=[], outputs=message
    )
    
block.launch(debug=True, share=True)
