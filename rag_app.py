
import os
import tempfile
import hashlib

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# --------------------------------------------------
# STEP 1: STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Multi-PDF RAG App",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF RAG Application")
st.write(
    "Upload multiple PDF documents and ask questions "
    "across all of them."
)


# --------------------------------------------------
# STEP 2: CHECK OPENAI API KEY
# --------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OPENAI_API_KEY is not configured. "
        "Please set your API key in CMD."
    )
    st.stop()


# --------------------------------------------------
# STEP 3: INITIALIZE SESSION STATE
# --------------------------------------------------

if "store" not in st.session_state:
    st.session_state.store = None

if "indexed_signature" not in st.session_state:
    st.session_state.indexed_signature = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# STEP 4: PDF UPLOAD
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


# Create a signature based on uploaded filenames
# and their contents.

def get_upload_signature(files):

    file_hashes = []

    for file in files:
        content = file.getvalue()

        file_hash = hashlib.md5(content).hexdigest()

        file_hashes.append(
            f"{file.name}:{file_hash}"
        )

    return hashlib.md5(
        "|".join(file_hashes).encode()
    ).hexdigest()


current_signature = None

if uploaded_files:
    current_signature = get_upload_signature(
        uploaded_files
    )


# --------------------------------------------------
# STEP 5: CHECK WHETHER INDEX IS CURRENT
# --------------------------------------------------

index_is_current = (
    st.session_state.store is not None
    and current_signature
    == st.session_state.indexed_signature
)

if (
    st.session_state.store is not None
    and not index_is_current
):
    st.warning(
        "Your uploaded PDFs have changed. "
        "Please build the index again."
    )


# --------------------------------------------------
# STEP 6: CREATE EMBEDDINGS AND INDEX PDFs
# --------------------------------------------------

if st.button(
    "Build / Rebuild PDF Index",
    type="primary"
):

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")

    else:

        all_chunks = []

        # Split PDF text into smaller chunks.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        progress = st.progress(0)

        total_files = len(uploaded_files)

        for file_number, uploaded_file in enumerate(
            uploaded_files
        ):

            st.write(
                f"Processing: {uploaded_file.name}"
            )

            temporary_path = None

            try:

                # Save uploaded PDF temporarily.
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getvalue()
                    )

                    temporary_path = temp_file.name

                # Load PDF pages.
                loader = PyPDFLoader(
                    temporary_path
                )

                pages = loader.load()

                # Replace temporary path with
                # original uploaded filename.
                for page in pages:

                    page.metadata["source"] = (
                        uploaded_file.name
                    )

                # Split pages into chunks.
                chunks = text_splitter.split_documents(
                    pages
                )

                all_chunks.extend(chunks)

            finally:

                # Remove temporary PDF.
                if (
                    temporary_path
                    and os.path.exists(temporary_path)
                ):
                    os.remove(temporary_path)

            progress.progress(
                (file_number + 1) / total_files
            )

        if not all_chunks:

            st.error(
                "No readable text was found "
                "in the uploaded PDFs."
            )

        else:

            # Create OpenAI embeddings.
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key
            )

            # Put all chunks into ONE FAISS store.
            vector_store = FAISS.from_documents(
                all_chunks,
                embeddings
            )

            # Save the vector store in session state.
            st.session_state.store = vector_store

            st.session_state.indexed_signature = (
                current_signature
            )

            st.session_state.chunk_count = len(
                all_chunks
            )

            # Clear old conversation after reindexing.
            st.session_state.messages = []

            st.success(
                f"Successfully indexed "
                f"{len(all_chunks)} chunks "
                f"from {total_files} PDFs."
            )

# Refresh index status after building the index.
index_is_current = (
    st.session_state.store is not None
    and current_signature
    == st.session_state.indexed_signature
)
# --------------------------------------------------
# STEP 7: DISPLAY INDEX INFORMATION
# --------------------------------------------------

if st.session_state.store is not None:

    st.info(
        f"Total indexed chunks: "
        f"{st.session_state.chunk_count}"
    )


# --------------------------------------------------
# STEP 8: DISPLAY PREVIOUS CHAT MESSAGES
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            with st.expander("📄 Sources"):

                for source in message.get(
                    "sources", []
                ):

                    st.caption(source)


# --------------------------------------------------
# STEP 9: QUESTION INPUT
# --------------------------------------------------

if index_is_current:

    question = st.chat_input(
        "Ask a question about your PDFs..."
    )

else:

    question = None

    st.info(
        "Upload your PDFs and build the index "
        "before asking questions."
    )


# --------------------------------------------------
# STEP 10: RETRIEVE RELEVANT DOCUMENT CHUNKS
# --------------------------------------------------

if question:

    # Display user question.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching PDFs and generating answer..."
        ):

            # Retrieve relevant chunks.
            retriever = (
                st.session_state.store
                .as_retriever(
                    search_kwargs={"k": 5}
                )
            )

            retrieved_docs = retriever.invoke(
                question
            )


            # --------------------------------------------------
            # STEP 11: PREPARE CONTEXT
            # --------------------------------------------------

            context_parts = []

            source_list = []

            for doc in retrieved_docs:

                filename = doc.metadata.get(
                    "source",
                    "Unknown file"
                )

                # PyPDFLoader page metadata
                # starts from zero.
                page_number = (
                    doc.metadata.get("page", 0) + 1
                )

                context_parts.append(
                    f"Source: {filename}, "
                    f"Page: {page_number}\n"
                    f"{doc.page_content}"
                )

                source_text = (
                    f"{filename} - Page {page_number}"
                )

                if source_text not in source_list:
                    source_list.append(
                        source_text
                    )

            context = "\n\n".join(
                context_parts
            )


            # --------------------------------------------------
            # STEP 12: CREATE LLM
            # --------------------------------------------------

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=api_key
            )


            # --------------------------------------------------
            # STEP 13: GENERATE ANSWER
            # --------------------------------------------------

            prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the
information provided in the retrieved context.

Instructions:
1. Do not invent information.
2. If the answer is not available in the
   context, say that the uploaded documents
   do not contain enough information.
3. If documents disagree, clearly explain
   the disagreement instead of choosing
   one answer silently.
4. Mention the relevant document names
   and page numbers in your answer.
5. Be clear and concise.

Retrieved context:

{context}

User question:

{question}

Answer:
"""

            response = llm.invoke(prompt)

            answer = response.content


            # --------------------------------------------------
            # STEP 14: DISPLAY ANSWER
            # --------------------------------------------------

            st.markdown(answer)


            # --------------------------------------------------
            # STEP 15: DISPLAY SOURCES
            # --------------------------------------------------

            with st.expander(
                "📄 View source documents and pages",
                expanded=True
            ):

                for source in source_list:

                    st.caption(source)


            # --------------------------------------------------
            # STEP 16: SAVE ANSWER IN CHAT HISTORY
            # --------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": source_list
                }
            )
