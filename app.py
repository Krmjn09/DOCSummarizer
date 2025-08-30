import streamlit as st
import os
from google.cloud import aiplatform
import google.generativeai as genai
from google.cloud import vision
from google.oauth2 import service_account
import PyPDF2
import docx
import io
import json
import base64
import fitz  # PyMuPDF for better PDF handling
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LegalDocumentAI:
    def __init__(self):
        self.setup_google_cloud()
        
    def setup_google_cloud(self):
        """Initialize Google Cloud services"""
        try:
            # Load service account key
            if os.path.exists("service-account-key.json"):
                credentials = service_account.Credentials.from_service_account_file(
                    "service-account-key.json"
                )
                
                # Configure Gemini
                genai.configure(credentials=credentials)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Initialize Vertex AI
                with open("service-account-key.json") as f:
                    key_data = json.load(f)
                    project_id = key_data.get("project_id")
                
                aiplatform.init(
                    project=project_id,
                    location="us-central1",
                    credentials=credentials
                )
                
                st.success("✅ AI services connected successfully!")
                return True
                
        except Exception as e:
            st.error(f"❌ Error connecting to AI services: {str(e)}")
            return False
    
    def extract_text_from_image(self, image_bytes):
        """Extract text from image using Tesseract OCR"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert PIL to numpy array for OpenCV
            img_array = np.array(image)
            
            # Improve image quality for better OCR
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply threshold to get better contrast
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Use Tesseract to extract text
            text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')
            
            return text.strip()
            
        except Exception as e:
            st.warning(f"OCR failed: {str(e)}")
            # Fallback: try without image processing
            try:
                image = Image.open(io.BytesIO(image_bytes))
                text = pytesseract.image_to_string(image)
                return text.strip()
            except:
                return ""
    
    def extract_text_from_file(self, uploaded_file):
        """Extract text from uploaded document with OCR support"""
        text = ""
        
        try:
            if uploaded_file.type == "application/pdf":
                # Try PyMuPDF first (better for complex PDFs)
                try:
                    file_bytes = uploaded_file.read()
                    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
                    
                    for page_num in range(pdf_document.page_count):
                        page = pdf_document[page_num]
                        
                        # First try to extract text directly
                        page_text = page.get_text()
                        
                        # If no text or very little text, use OCR on images
                        if len(page_text.strip()) < 50:
                            # Get page as image
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
                            img_bytes = pix.tobytes("png")
                            
                            # Use OCR to extract text from image
                            ocr_text = self.extract_text_from_image(img_bytes)
                            if ocr_text:
                                text += f"\n--- Page {page_num + 1} (OCR) ---\n{ocr_text}\n"
                            else:
                                text += page_text
                        else:
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
                
            elif uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
                # Handle image files directly
                image_bytes = uploaded_file.read()
                text = self.extract_text_from_image(image_bytes)
                if not text:
                    text = "No text found in the image."
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return ""
            
        return text.strip()
    
    def simplify_legal_text(self, text, analysis_type="summary"):
        """Use AI to simplify legal document"""
        
        prompts = {
            "summary": f"""
            You are a legal expert who explains complex legal documents in simple terms that anyone can understand.
            
            Please analyze this legal document and provide:
            1. A simple summary in everyday language (as if explaining to a friend)
            2. Key points that the person should know
            3. Any potential risks or important clauses
            4. Action items or things to watch out for
            
            Legal Document:
            {text}
            
            Please respond in a friendly, clear manner using simple words and short sentences.
            """,
            
            "risks": f"""
            You are a legal advisor helping someone understand potential risks in a legal document.
            
            Analyze this document and identify:
            1. Financial risks (extra fees, penalties, costs)
            2. Legal obligations (what you MUST do)
            3. Things you're giving up (rights you're waiving)
            4. Consequences of breaking the agreement
            5. Red flags or unusual clauses
            
            Legal Document:
            {text}
            
            Explain each risk clearly and suggest what to watch out for.
            """,
            
            "questions": f"""
            You are helping someone prepare questions to ask before signing this legal document.
            
            Based on this document, suggest important questions they should ask:
            1. Questions about costs and fees
            2. Questions about responsibilities and obligations
            3. Questions about what happens if things go wrong
            4. Questions about cancellation or changes
            5. Questions about unclear terms
            
            Legal Document:
            {text}
            
            Provide specific, practical questions they can ask.
            """
        }
        
        try:
            prompt = prompts.get(analysis_type, prompts["summary"])
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error analyzing document: {str(e)}"
    
    def analyze_specific_clause(self, text, user_question):
        """Answer specific questions about the document"""
        prompt = f"""
        You are a legal expert helping someone understand a specific part of their legal document.
        
        Document: {text}
        
        Question: {user_question}
        
        Please provide a clear, simple explanation that directly answers their question.
        Use everyday language and give practical advice.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    st.set_page_config(
        page_title="Legal Document AI Assistant", 
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Legal Document AI Assistant")
    st.markdown("**Upload any legal document and get simple, clear explanations in seconds!**")
    
    # Initialize the AI system
    if 'legal_ai' not in st.session_state:
        st.session_state.legal_ai = LegalDocumentAI()
    
    # Sidebar for navigation
    st.sidebar.title("📋 How to Use")
    st.sidebar.markdown("""
    1. **Upload** your legal document (PDF, Word, or Text)
    2. **Choose** what type of analysis you want
    3. **Get** simple explanations in plain English
    4. **Ask** specific questions about clauses
    """)
    
    # File upload
    st.header("📄 Upload Your Document")
    uploaded_file = st.file_uploader(
        "Choose your legal document", 
        type=['pdf', 'docx', 'txt'],
        help="Support for rental agreements, contracts, terms of service, etc."
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        # Extract text
        with st.spinner("📖 Reading your document..."):
            document_text = st.session_state.legal_ai.extract_text_from_file(uploaded_file)
        
        if document_text.strip():
            # Show document preview
            with st.expander("📖 Document Preview"):
                st.text_area("Document Content", document_text[:1000] + "..." if len(document_text) > 1000 else document_text, height=200)
            
            # Analysis options
            st.header("🤖 AI Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📝 Get Simple Summary", use_container_width=True):
                    with st.spinner("🧠 AI is analyzing your document..."):
                        summary = st.session_state.legal_ai.simplify_legal_text(document_text, "summary")
                    st.session_state.current_analysis = summary
            
            with col2:
                if st.button("⚠️ Find Risks & Red Flags", use_container_width=True):
                    with st.spinner("🔍 Identifying potential risks..."):
                        risks = st.session_state.legal_ai.simplify_legal_text(document_text, "risks")
                    st.session_state.current_analysis = risks
            
            with col3:
                if st.button("❓ Questions to Ask", use_container_width=True):
                    with st.spinner("💡 Preparing questions for you..."):
                        questions = st.session_state.legal_ai.simplify_legal_text(document_text, "questions")
                    st.session_state.current_analysis = questions
            
            # Display analysis results
            if 'current_analysis' in st.session_state:
                st.header("📋 Analysis Results")
                st.markdown(st.session_state.current_analysis)
            
            # Q&A section
            st.header("💬 Ask Specific Questions")
            user_question = st.text_input("Ask anything about your document:", 
                                        placeholder="e.g., What happens if I break this contract?")
            
            if st.button("Get Answer") and user_question:
                with st.spinner("🤔 Thinking about your question..."):
                    answer = st.session_state.legal_ai.analyze_specific_clause(document_text, user_question)
                st.markdown(f"**Answer:** {answer}")
        
        else:
            st.error("❌ Could not extract text from the document. Please try a different file.")
    
    else:
        st.info("👆 Please upload a legal document to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("**⚡ Powered by Google Cloud AI | Built for the Legal AI Hackathon**")

if __name__ == "__main__":
    main()