# Copyright 2024 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import streamlit as st  # pip install streamlit
import base64
import vertexai  # pip install vertexai
import io
from PIL import Image
import PIL.Image
from vertexai.preview.generative_models import GenerativeModel, Part, GenerationConfig
import vertexai.preview.generative_models as generative_models
import tempfile
import google.auth
from google.auth import transport
from central_config import logo_path,brand_names,logo_width, gs_folder, company, styling
# Define the model that will be used to process your text
model = GenerativeModel("gemini-1.5-pro")  

########################################
# Functions
########################################

def keyload():  # Keyloader allows local dev with a key.json file, but falls back to google auth if you dont generate keys, or when it is deployed.
    print("centralconfig keyload")
    try:
        print("try keyload from file")
        # Load credentials from a local key file
        credentials, project_id = google.auth.load_credentials_from_file('../keys/key2.json')
        print("file loaded")
        return credentials, project_id
    except:
        print("command line auth?")
        # Fallback to command-line authentication
        credentials, project_id = google.auth.default()
        return credentials, project_id
credentials, project_id = keyload()

def vertex_upload_and_analyze_image(uploaded_file, image_prompt, image_analysis_output): # This function sends an image, a prompt of what to do, and a secondary prompt that tells Gemini how to respond.
    # You can also tweak the generation config (look it up!) to send a schema and have Gemini send back JSON instead of text.
    # Combine the prompts to instruct Gemini on the task and desired output format
    image_prompt = f"""
        Task: Analyze this image and follow these instructions {image_prompt}. 
        Task: Your output should be {image_analysis_output}. 
        If you are not confident in an answer, return Unknown
    """
    # Get the image content (directly from the in-memory file object)
    image_content = uploaded_file.getvalue()
    # Convert to Vertex required base64
    image1 = base64.b64encode(image_content).decode('utf-8')  
    # Use 'Part' function from Vertex AI to prepare the image
    image_ready = Part.from_data(image1, mime_type="image/jpeg")  
    # Send prompt and image to the model
    response = model.generate_content([image_prompt, image_ready])  
    return response.text


########################################
# The User Interface starts here
########################################
st.set_page_config(layout="wide")
st.markdown(f"""{styling}""",unsafe_allow_html=True)
row0_0, row0_1, row0_2 = st.columns((1,12,1))
with row0_1:
    st.markdown(
        f"""
        <div style="text-align: center;padding-bottom:0px;margin-top:-50px;">  
            <img src="{logo_path}" width="{logo_width}">
        </div>
       <div style="text-align: center;margin-top:-10px">  
            <h1>Image Analysis with Vertex</h1> 
            </br>
        </div>
        """,
        unsafe_allow_html=True
    )

with st.container(border=True):
    row0_0, row0_1, row0_2 = st.columns((1, 12, 1))

    # File uploader for the user to select an image
    uploaded_file = st.file_uploader("Choose an image JPG or PNG", type=["jpg", "jpeg", "png"])  
    go_button = st.button("Go")  # Button to trigger the analysis

# Define the prompt for image analysis
imageprompt = """
General Information
What objects are present in the image? (Identify the main objects and their categories)
What is the overall scene depicted in the image? (Describe the environment and context)
Are there any people in the image? If so, who are they and what are they doing? (Identify individuals, their actions, and potential relationships)
What is the emotional impact of the image? (Analyze the mood, tone, and feelings evoked)
What are the dominant colors in the image and what is their effect? (Analyze the color palette and its influence on the overall impression)
Composition and Technique
How is the image composed? (Analyze the arrangement of elements, leading lines, and visual balance)
What photographic or artistic techniques are used? (Identify techniques like lighting, perspective, depth of field, etc.)
Is there any text in the image? What does it say? (Extract and analyze any text present)
What is the resolution and aspect ratio of the image? (Get technical details about the image dimensions)
What is the potential story behind the image? (Encourage narrative interpretation)
What is the cultural or historical context of the image? (Analyze potential influences and references)
What symbols or metaphors are present? (Identify and interpret symbolic elements)
What message or idea is the image trying to convey? (Analyze the intended communication)
How does this image compare to others in a similar category or style? (Draw comparisons and analyze differences)
Image Quality
Is the image clear or blurry? (Assess the overall image quality)
These questions can be adapted and expanded upon depending on the specific focus of your image analysis application. You can also combine them with user-provided information or context to generate even more insightful analysis.
"""

if go_button:  # Check if the "Go" button is clicked
    with st.spinner(text="Analyzing Image..."):  # Display a spinner while processing
        with st.container(border=False):
            # Instructions for the desired output format from Gemini
            image_analysis_output = """
            Output is Markdown. Return multiple tables of analysis, 
            separate each table by an h4 sized header explaining what the data is, 
            provide a summary text at the end with your reasoning.  
            Example Output:
            A table with metadata from the receipt, the McDonald's store information. key/value. date, time, order number, address.
            A table with the breakdown of the Order information from the receipt. qty, name, NO, ADD.
            A table of data analyzing the food from the picture. Item, toppings included, toppings missing.
            Summary: written explanation of pass/fail, be verbose, use all data from the receipt and image for your justification.  
            If multiple issues are present be sure to talk to them.
            Do not use <br> HTML in your response, if you need to separate data just use a comma.      
            """
            if uploaded_file is None:  # Use a default image if no file is uploaded
                image_path = "./imgs/mcd-receipt.jpg"  
            else:
                # Create a temporary file to store the uploaded image
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:  
                    temp_file.write(uploaded_file.getbuffer())
                    image_path = temp_file.name
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                uploaded_file = io.BytesIO(image_data)

                row1_0, row1_1 = st.columns((6, 6))  # Create two columns for image and analysis
                with row1_0:
                    # Display the uploaded image
                    #st.image(uploaded_file, width=300)  #this code works
                    image_b64 = base64.b64encode(uploaded_file.getvalue()).decode() # a little bit of image conversion magic to get the local image as well as uploaded images to 'fit' at 100% in CSS.
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: center;">
                            <img src="data:image/jpeg;base64,{image_b64}" style="width: 100%; "> 
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    

                # Call the function to analyze the image and get the output
                output = vertex_upload_and_analyze_image(uploaded_file, imageprompt, image_analysis_output)  
                with row1_1:
                    # Display the analysis output in markdown format
                    st.markdown(output)