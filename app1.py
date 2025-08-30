import streamlit as st
import os
from google.cloud import aiplatform
import google.generativeai as genai
from google.oauth2 import service_account
import PyPDF2
import docx
import io
import json
import fitz  # PyMuPDF for better PDF handling
# from dotenv import load_dotenv
import json
import streamlit as st
from google.oauth2 import service_account

# Load environment variables
# load_dotenv()

class LegalDocumentAI:
    def __init__(self):
        self.setup_google_cloud()
    
    def setup_google_cloud(self):
        """Initialize Google Cloud services"""
        try:
            if "GCP_SERVICE_ACCOUNT" in st.secrets:
                key_dict = json.loads(st.secrets["GCP_SERVICE_ACCOUNT"])
                credentials = service_account.Credentials.from_service_account_info(key_dict)
    
                # Configure Gemini
                genai.configure(credentials=credentials)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
    
                # Initialize Vertex AI (use key_dict directly, no file read)
                aiplatform.init(
                    project=key_dict["project_id"],
                    location="us-central1",
                    credentials=credentials
                )
    
                st.success("✅ AI services connected successfully!")
                return True
    
        except Exception as e:
            st.error(f"❌ Error connecting to AI services: {str(e)}")
            return False
    
    def extract_text_from_file(self, uploaded_file):
        """Extract text from uploaded document (text-based files only)"""
        text = ""
        
        try:
            if uploaded_file.type == "application/pdf":
                # Try PyMuPDF first (better for complex PDFs)
                try:
                    file_bytes = uploaded_file.read()
                    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
                    
                    for page_num in range(pdf_document.page_count):
                        page = pdf_document[page_num]
                        
                        # Extract text directly from PDF
                        page_text = page.get_text()
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    
                    pdf_document.close()
                    
                except Exception as e:
                    st.warning(f"PyMuPDF failed, trying PyPDF2: {str(e)}")
                    # Fallback to PyPDF2
                    uploaded_file.seek(0)  # Reset file pointer
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                        
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                # Handle Word documents
                doc = docx.Document(io.BytesIO(uploaded_file.read()))
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                    
            elif uploaded_file.type == "text/plain":
                # Handle text files
                text = uploaded_file.read().decode("utf-8")
                
            else:
                st.error(f"Unsupported file type: {uploaded_file.type}")
                return ""
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return ""
            
        return text.strip()
    
    def simplify_legal_text(self, text, analysis_type="summary"):
        """Use AI to simplify legal document with finance-focused prompts"""
        
        prompts = {
            "summary": f"""
            You are a financial and legal expert who specializes in explaining complex legal documents that impact people's money and financial well-being.
            
            Please analyze this legal document and provide:
            1. **Plain English Summary**: What is this document about? (as if explaining to someone with no legal background)
            2. **Financial Impact**: How will this affect the person's money, assets, or financial obligations?
            3. **Key Terms & Conditions**: What are the most important rules they need to follow?
            4. **Cost Breakdown**: All fees, charges, penalties, and potential additional costs
            5. **Your Rights**: What protections and rights do you have?
            6. **Red Flags**: Any concerning clauses that could lead to financial loss
            
            Legal Document:
            {text}
            
            Please use simple language and focus on the financial implications. Highlight any terms that could cost money or create financial obligations.
            """,
            
            "risks": f"""
            You are a financial advisor helping someone understand the financial and legal risks in this document.
            
            Analyze this document and identify:
            1. **Financial Risks**: 
               - Hidden fees, charges, or penalties
               - Variable costs that could increase
               - Situations where you might owe money
               - Impact on credit score or financial standing
            
            2. **Legal Obligations**: 
               - What you MUST do (and what it costs if you don't)
               - Deadlines and time-sensitive requirements
               - Automatic renewals or extensions
            
            3. **Rights You're Giving Up**: 
               - Ability to sue or seek damages
               - Privacy rights with your data
               - Freedom to choose alternatives
            
            4. **Exit Costs**: 
               - Cancellation fees or penalties
               - Early termination costs
               - What happens to deposits or payments made
            
            5. **Warning Signs**: 
               - Unusual or one-sided clauses
               - Terms that seem too good to be true
               - Vague language around costs
            
            Legal Document:
            {text}
            
            For each risk, explain the potential financial impact and provide practical advice on how to protect yourself.
            """,
            
            "questions": f"""
            You are helping someone prepare smart questions to ask before signing this legal document to protect their financial interests.
            
            Based on this document, here are the important questions they should ask:
            
            **About Money & Costs:**
            1. What are ALL the fees involved? (setup, monthly, annual, hidden charges)
            2. Can costs increase over time? By how much and how often?
            3. What triggers penalty fees and how much are they?
            4. Are there any situations where I might owe additional money?
            
            **About Terms & Flexibility:**
            5. Can I cancel this agreement? What does it cost to get out?
            6. What happens if I miss a payment or deadline?
            7. Does this automatically renew? How do I opt out?
            8. Can you change the terms without my agreement?
            
            **About Protection & Rights:**
            9. What happens if you don't deliver what's promised?
            10. Do I have any recourse if there's a dispute?
            11. Is my personal/financial information protected?
            12. What are my rights if something goes wrong?
            
            **About Specific Clauses:**
            [Based on the document content, add 3-5 specific questions about unclear or concerning terms]
            
            Legal Document:
            {text}
            
            Provide specific questions tailored to this document that will help them make an informed financial decision.
            """
        }
        
        try:
            prompt = prompts.get(analysis_type, prompts["summary"])
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error analyzing document: {str(e)}"
    
    def analyze_specific_clause(self, text, user_question):
        """Answer specific questions about the document with financial focus"""
        prompt = f"""
        You are a financial and legal expert helping someone understand their legal document, particularly focusing on financial implications.
        
        Document: {text}
        
        Question: {user_question}
        
        Please provide a clear, practical answer that:
        1. Directly answers their question in simple terms
        2. Explains any financial impact or cost implications
        3. Highlights risks or benefits they should know about
        4. Suggests practical next steps or things to watch out for
        
        Focus on how this affects their money, rights, and financial security.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    st.set_page_config(
        page_title="Legal Document AI - Demystify Legal Jargon", 
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 Legal Document AI - Financial Protection Assistant")
    st.markdown("**Demystify complex legal documents and protect your financial interests! Upload any legal document and get clear explanations about costs, risks, and your rights.**")
    
    # Add a banner about the purpose
    st.info("🎯 **Specialized for Financial Protection**: This tool focuses on the financial implications of legal documents - helping you understand costs, risks, and protect your money!")
    
    # Initialize the AI system
    if 'legal_ai' not in st.session_state:
        st.session_state.legal_ai = LegalDocumentAI()
    
    # Sidebar for navigation
    st.sidebar.title("📋 How to Use")
    st.sidebar.markdown("""
    1. **Upload** your legal document (PDF, Word, or Text file)
    2. **Choose** what type of analysis you want:
       - 💡 **Simple Summary**: Plain English explanation
       - ⚠️ **Financial Risks**: Money-related dangers
       - ❓ **Smart Questions**: What to ask before signing
    3. **Get** clear explanations focused on financial impact
    4. **Ask** specific questions about costs, terms, or risks
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔍 What We Analyze:**")
    st.sidebar.markdown("""
    - Rental & Lease Agreements
    - Loan Contracts & Mortgages
    - Employment Contracts
    - Service Terms & Conditions
    - Insurance Policies
    - Purchase Agreements
    - Any legal document with financial terms!
    """)
    
    # File upload
    st.header("📄 Upload Your Legal Document")
    uploaded_file = st.file_uploader(
        "Choose your legal document", 
        type=['pdf', 'docx', 'txt'],
        help="Supports PDF, Word docs, and text files with readable text"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        # Extract text
        with st.spinner("📖 Reading and analyzing your document..."):
            document_text = st.session_state.legal_ai.extract_text_from_file(uploaded_file)
        
        if document_text.strip():
            # Show document preview
            with st.expander("📖 Document Preview (Click to expand)"):
                st.text_area("Document Content", document_text[:1500] + "..." if len(document_text) > 1500 else document_text, height=200)
            
            # Analysis options
            st.header("🤖 AI Financial Analysis")
            st.markdown("Choose the type of analysis that best fits your needs:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💡 Get Simple Summary", use_container_width=True, help="Plain English explanation focusing on financial impact"):
                    with st.spinner("🧠 AI is analyzing your document for financial implications..."):
                        summary = st.session_state.legal_ai.simplify_legal_text(document_text, "summary")
                    st.session_state.current_analysis = summary
                    st.session_state.analysis_type = "Summary"
            
            with col2:
                if st.button("⚠️ Find Financial Risks", use_container_width=True, help="Identify potential costs, penalties, and financial dangers"):
                    with st.spinner("🔍 Identifying financial risks and red flags..."):
                        risks = st.session_state.legal_ai.simplify_legal_text(document_text, "risks")
                    st.session_state.current_analysis = risks
                    st.session_state.analysis_type = "Financial Risks"
            
            with col3:
                if st.button("❓ Smart Questions to Ask", use_container_width=True, help="Get prepared questions to ask before signing"):
                    with st.spinner("💡 Preparing smart questions to protect your interests..."):
                        questions = st.session_state.legal_ai.simplify_legal_text(document_text, "questions")
                    st.session_state.current_analysis = questions
                    st.session_state.analysis_type = "Questions to Ask"
            
            # Display analysis results
            if 'current_analysis' in st.session_state:
                st.header(f"📋 {st.session_state.get('analysis_type', 'Analysis')} Results")
                st.markdown(st.session_state.current_analysis)
                
                # Add download button for the analysis
                st.download_button(
                    label="📄 Download Analysis",
                    data=st.session_state.current_analysis,
                    file_name=f"legal_analysis_{uploaded_file.name}.txt",
                    mime="text/plain"
                )
            
            # Q&A section
            st.header("💬 Ask Specific Questions")
            st.markdown("Got a specific concern? Ask anything about costs, terms, risks, or your rights:")
            
            # Provide example questions
            example_questions = [
                "What happens if I can't pay on time?",
                "What are all the fees I need to pay?",
                "Can I cancel this agreement?",
                "What am I liable for if something goes wrong?",
                "Are there any hidden costs?",
                "What rights am I giving up?"
            ]
            
            selected_example = st.selectbox("Or choose from common questions:", [""] + example_questions)
            
            user_question = st.text_input("Your question:", 
                                        value=selected_example,
                                        placeholder="e.g., What happens if I break this contract? What are the cancellation fees?")
            
            if st.button("Get Answer") and user_question:
                with st.spinner("🤔 Analyzing your specific question..."):
                    answer = st.session_state.legal_ai.analyze_specific_clause(document_text, user_question)
                
                st.markdown("### 💡 Answer:")
                st.markdown(answer)
        
        else:
            st.error("❌ Could not extract text from the document. Please ensure the file contains readable text or try a different file format.")
    
    else:
        # Welcome section with examples
        st.markdown("## 🚀 Get Started")
        st.markdown("Upload any legal document to get instant, clear explanations focused on protecting your financial interests!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Supported Documents:")
            st.markdown("""
            - **Rental Agreements** - Hidden fees, deposit terms, penalties
            - **Loan Contracts** - Interest rates, fees, payment terms
            - **Employment Contracts** - Salary, benefits, termination clauses
            - **Service Agreements** - Costs, cancellation policies, liability
            - **Terms of Service** - What you're agreeing to financially
            """)
        
        with col2:
            st.markdown("### 💡 What You'll Learn:")
            st.markdown("""
            - **All Costs & Fees** - Upfront, ongoing, and hidden charges
            - **Financial Risks** - What could cost you money
            - **Your Rights** - What protections you have
            - **Exit Strategy** - How to get out and what it costs
            - **Red Flags** - Warning signs to watch for
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("**⚡ Powered by Google Cloud AI | Built for Legal Document Demystification**")
    st.markdown("*💡 This tool provides educational information only. For legal advice, consult a qualified attorney.*")

if __name__ == "__main__":
    main()