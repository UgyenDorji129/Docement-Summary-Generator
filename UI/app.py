import streamlit as st
import requests


def create_landing_page(paths):
    #Giving the title of the Landing Page
    st.title('ASK YOUR OWN DOCUMENTS 👻')
    option = st.selectbox(
        'Select which documents you want to summarise?',
        ("None", 'PDF', 'Docx', 'TXT'))
    
    #If none is selected, API is not hit.
    if (option != "None"):
        url = "http://127.0.0.1:5000/summarize_doc"

        data = {
            "path": paths[option]
        }
        try:
            response = requests.post(url, json=data)
            value = response.json()
            st.write('Your summary:')
            st.write(value['result'])
        except:
            st.write('Extremly Sorry!!! Some unexpected error occured')

if __name__ == "__main__":
    # Hardcoded the path map to the document type
    paths = {
        "PDF": "/home/udorji/Documents/TRACK-3/generative-ai-assignment-ugyen505dorji-1686737015613/assets/TheTurtleandtheRabbitFiction3rdGrade.pdf",
        "Docx": "/home/udorji/Documents/TRACK-3/generative-ai-assignment-ugyen505dorji-1686737015613/assets/Document.docx",
        "TXT": "/home/udorji/Documents/TRACK-3/generative-ai-assignment-ugyen505dorji-1686737015613/assets/big.txt"
    }
    create_landing_page(paths)
