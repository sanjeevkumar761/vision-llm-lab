import os  
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI  
import base64
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("OPENAI_API_BASE")  
deployment = os.getenv("GPT4O_MODEL_DEPLOYMENT_NAME")  
subscription_key = os.getenv("OPENAI_API_KEY")  
document_intelligence_endpoint = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
document_intelligence_key = os.getenv("DOC_INTELLIGENCE_KEY", )

# Initialize Azure OpenAI client with key-based authentication
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2024-05-01-preview",  
)  

# Initialize Document Intelligence client
document_intelligence_client = DocumentIntelligenceClient(
    endpoint=document_intelligence_endpoint,
    credential=AzureKeyCredential(document_intelligence_key),
    api_version="2023-10-31-preview"
)

def extract_markdown_from_page(pdf_path, page_number):
    with open(pdf_path, "rb") as f:
        document_content = f.read()
        base64_document = base64.b64encode(document_content).decode('utf-8')
        poller = document_intelligence_client.begin_analyze_document(
            model_id="prebuilt-layout", 
            analyze_request={"base64Source": base64_document},
            output_content_format="markdown"
        )
        result = poller.result()



    extracted_text = ""
    for page in result.pages:
        if page.page_number == page_number:
            for line in page.lines:
                extracted_text += line.content + "\n"
            break  # Exit the loop once the desired page is processed

    return extracted_text

# Example usage
pdf_path = "file.pdf"
page_number = 1  # Specify the page number you want to extract
page_text = extract_markdown_from_page(pdf_path, page_number)
print(page_text)

# Prepare the prompt  
prompt = [
    {
        "role": "system",
        "content": "You are an AI assistant that helps extract details from a document. Also show the page number."
    },
    {
        "role": "user",
        "content": "Provide granular, micro level details from the document given here. " + f"Here is is the document: {page_text}"  # Replace with the actual user prompt
    }
]  

# Generate the completion  
completion = client.chat.completions.create(
    messages=prompt,
    model=deployment  # Assuming the deployment name is the model name
)

# Access the content from the completion object
content = completion.choices[0].message.content
print(content)