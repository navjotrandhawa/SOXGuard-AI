import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import BaseModel, Field
import tempfile
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Load environment variables
load_dotenv()

# ==========================================
# 1. Define Structured Output Schema (Pydantic with Relevance Guardrail)
# ==========================================
class ComplianceSWOT(BaseModel):
    is_relevant_document: bool = Field(
        description="True ONLY if the uploaded documents are valid business requirements (BRDs), SEC 10-Ks, financial reports, or internal control frameworks. False if they are irrelevant, personal files (like astrology kundlis, horoscopes, resumes), random text, or gibberish."
    )
    alignment_status: str = Field(
        description="Whether the documents align with the standard SOX baseline: 'Aligned', 'Partially Aligned', 'Not Aligned', or 'Invalid Document Type'."
    )
    rating: int = Field(
        description="Rating out of 10, where 10 is perfect alignment and 0 is complete failure or invalid/irrelevant document."
    )
    alignment_summary: str = Field(
        description="Detailed summary explaining why the rating and status were given based on the documents, or explaining why an uploaded file failed relevance validation."
    )
    strengths: list[str] = Field(description="List of strengths identified in the client documents relative to SOX baseline, or empty if invalid.")
    weaknesses: list[str] = Field(description="List of weaknesses or gaps found, or reasons why the document failed relevance.")
    opportunities: list[str] = Field(description="Opportunities for process or control improvement, or empty.")
    threats: list[str] = Field(description="Potential audit risks or compliance threats, or empty.")

# ==========================================
# 2. PDF Report Generator Helper
# ==========================================
def generate_pdf_report(result, audit_focus):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title & Metadata
    story.append(Paragraph("SOX Compliance Audit Report", styles['Title']))
    story.append(Paragraph(f"<b>Audit Focus:</b> {audit_focus}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Status & Rating
    story.append(Paragraph(f"<b>Alignment Status:</b> {result.alignment_status}", styles['Normal']))
    story.append(Paragraph(f"<b>Compliance Rating:</b> {result.rating} / 10", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Summary
    story.append(Paragraph("<b>Executive Summary</b>", styles['Heading2']))
    story.append(Paragraph(result.alignment_summary, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # SWOT Sections
    swot_sections = [
        ("Strengths", result.strengths),
        ("Weaknesses", result.weaknesses),
        ("Opportunities", result.opportunities),
        ("Threats & Risks", result.threats)
    ]
    
    for title, items in swot_sections:
        story.append(Paragraph(f"<b>{title}</b>", styles['Heading2']))
        if items:
            for item in items:
                story.append(Paragraph(f"• {item}", styles['Normal']))
        else:
            story.append(Paragraph("None identified.", styles['Normal']))
        story.append(Spacer(1, 8))
        
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ==========================================
# 3. Streamlit UI & Session State Setup
# ==========================================
st.set_page_config(page_title="SOX Compliance Auditor", layout="wide")
st.title("🛡️ AI-Powered SOX Compliance Auditor with Guardrails & Memory")
st.markdown("Upload compliance documents, select a custom audit focus, and chat interactively with context persistence.")

# Initialize session state for message history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
if "audit_run" not in st.session_state:
    st.session_state.audit_run = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None

with st.sidebar:
    st.header("📁 Document Management")
    
    # Disclaimer notice
    st.info("💡 **Disclaimer:** Please refresh the browser page to start a new session and upload a new set of documents.")
    
    st.markdown("---")
    uploaded_brd = st.file_uploader("Upload Client Business Requirements Document (BRD)", type=["pdf"])
    uploaded_sec = st.file_uploader("Upload SEC 10-K / Financial Extract", type=["pdf"])
    
    st.header("🎛️ Dynamic Audit Choices")
    audit_focus = st.selectbox(
        "Select Audit Focus Area:",
        [
            "Comprehensive Audit (Full SWOT & Baseline Check)",
            "Segregation of Duties (SoD) Focus",
            "Access Control & RBAC Focus",
            "Audit Trails & Documentation Rigor"
        ]
    )
    
    run_audit = st.button("Run Compliance Audit", type="primary")

# ==========================================
# 4. Core Processing & Evaluation Logic
# ==========================================
if run_audit:
    if not uploaded_brd or not uploaded_sec:
        st.error("Please upload both PDF documents to proceed.")
    else:
        with st.spinner("Processing documents, running relevance checks, and performing audit analysis..."):
            
            def save_temp_file(uploaded_file):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    return tmp_file.name

            brd_path = save_temp_file(uploaded_brd)
            sec_path = save_temp_file(uploaded_sec)

            # Load and chunk documents
            loader_brd = PyPDFLoader(brd_path)
            loader_sec = PyPDFLoader(sec_path)
            all_pages = loader_brd.load() + loader_sec.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(all_pages)

            embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

            retrieved_docs = retriever.invoke(f"SOX controls compliance financial reporting {audit_focus}")
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

            os.unlink(brd_path)
            os.unlink(sec_path)

            standard_sox_baseline = """
            Standard SOX Baseline Requirements:
            1. Section 404 Compliance: Management must establish and maintain an adequate internal control structure and financial reporting procedures.
            2. Segregation of Duties (SoD): Key financial processes must separate authorization, custody, and recordkeeping roles.
            3. Audit Trails & Documentation: All system modifications, financial data entries, and reconciliation logs must have immutable audit trails.
            4. Access Controls: Role-based access control (RBAC) must restrict financial system configuration and ledger postings strictly to authorized finance personnel.
            """

            parser = PydanticOutputParser(pydantic_object=ComplianceSWOT)

            # Dynamic Prompt Template incorporating relevance guardrail and chat memory placeholder
            prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "You are an expert, strict GRC and SOX Auditor specializing in {audit_focus}.\n\n"
                 "### CRITICAL RELEVANCE GATE INSTRUCTION:\n"
                 "Before performing any compliance audit, verify if the retrieved text from the uploaded client documents actually relates to financial reporting, business requirements (BRDs), SEC regulations, or internal controls (SOX).\n"
                 "- If the text contains personal records, horoscopes, astrology (Kundli), cooking recipes, resumes, or completely unrelated gibberish, you MUST set `is_relevant_document` to `false`, rating to `0`, alignment_status to 'Invalid Document Type', and explain in the summary that the file is invalid.\n"
                 "- Do NOT force irrelevant text into compliance categories.\n\n{format_instructions}"
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "### Standard SOX Baseline:\n{baseline}\n\n"
                          "### Retrieved Content from Uploaded Client Documents:\n{context}\n\n"
                          "Perform a rigorous compliance audit focusing on: {audit_focus}.")
            ])

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            chain = prompt | llm | parser

            chain_with_history = RunnableWithMessageHistory(
                chain,
                lambda session_id: st.session_state.chat_history,
                input_messages_key="context",
                history_messages_key="history",
            )

            try:
                result = chain_with_history.invoke(
                    {
                        "baseline": standard_sox_baseline,
                        "context": context_text,
                        "audit_focus": audit_focus,
                        "format_instructions": parser.get_format_instructions()
                    },
                    config={"configurable": {"session_id": "sox_audit_session"}}
                )

                st.session_state.last_result = result
                st.session_state.audit_run = True

            except Exception as e:
                st.error(f"An error occurred during chain execution: {e}")

# ==========================================
# 5. Display Results & Interactive Chat History
# ==========================================
if st.session_state.get("audit_run", False):
    result = st.session_state.last_result
    
    # Handle invalid or irrelevant document uploads gracefully
    if not result.is_relevant_document:
        st.error("🚨 **Invalid Document Detected by Relevance Gate!**")
        st.warning(result.alignment_summary)
        st.metric(label="Compliance Rating", value="0 / 10")
    else:
        st.success("Audit Completed Successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Alignment Status", value=result.alignment_status)
        with col2:
            st.metric(label="Compliance Rating", value=f"{result.rating} / 10")

        # PDF Download Button
        pdf_data = generate_pdf_report(result, audit_focus)
        st.download_button(
            label="📥 Download Audit Report (PDF)",
            data=pdf_data,
            file_name="SOX_Compliance_Audit_Report.pdf",
            mime="application/pdf"
        )

        st.markdown("### 📋 Executive Summary")
        st.write(result.alignment_summary)

        st.markdown("---")
        st.subheader("🔍 SWOT Analysis Matrix")

        swot_col1, swot_col2 = st.columns(2)
        
        with swot_col1:
            st.markdown("#### 💪 Strengths")
            for s in result.strengths:
                st.markdown(f"- {s}")

            st.markdown("#### 📈 Opportunities")
            for o in result.opportunities:
                st.markdown(f"- {o}")

        with swot_col2:
            st.markdown("#### ⚠️ Weaknesses")
            for w in result.weaknesses:
                st.markdown(f"- {w}")

            st.markdown("#### 🚨 Threats & Risks")
            for t in result.threats:
                st.markdown(f"- {t}")

    # ==========================================
    # 6. Interactive Chat & Memory Section
    # ==========================================
    st.markdown("---")
    st.subheader("💬 Ask Follow-up Questions (Powered by LLM Memory)")
    
    for msg in st.session_state.chat_history.messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

    if follow_up_query := st.chat_input("Ask a follow-up question about the audit or documents..."):
        with st.chat_message("user"):
            st.write(follow_up_query)

        st.session_state.chat_history.add_user_message(follow_up_query)
        
        with st.spinner("Thinking..."):
            chat_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful SOX compliance assistant answering follow-up queries based on previous audit findings."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            chat_chain = chat_prompt | chat_llm
            
            chat_chain_with_history = RunnableWithMessageHistory(
                chat_chain,
                lambda session_id: st.session_state.chat_history,
                input_messages_key="input",
                history_messages_key="history",
            )
            
            response = chat_chain_with_history.invoke(
                {"input": follow_up_query},
                config={"configurable": {"session_id": "sox_audit_session"}}
            )
            
            with st.chat_message("assistant"):
                st.write(response.content)


                
