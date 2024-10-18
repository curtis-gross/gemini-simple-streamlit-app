# Copyright 2024 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Import necessary libraries
import base64
import json
import os
import pandas as pd
import streamlit as st
import vertexai
from google.auth import transport

# Import custom functions and configurations from central_config.py
from central_config import (
    logo_path, logo_width, company, styling, vertex_text, df,
    vertex_text_strict, vertex_image
)

# Set Streamlit page configuration
st.set_page_config(layout="wide")
st.markdown(f"""{styling}""", unsafe_allow_html=True)

# --- Functions ---


def display_images(img_list, imgfilename):
    """Saves generated images to local storage."""
    z = 0
    data_dir = "./data"  # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)
    for each in img_list:
        img_path = os.path.join(data_dir, f"{z}-{imgfilename}.jpg")
        each.save(img_path)
        print(f"Image {z} saved to {img_path}")
        z += 1


def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Generates an HTML link to download a binary file."""
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href_link = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href_link

# --- Main App ---

# SQL parameters (if needed)
sql_parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "seed": 1,
    "temperature": 0.5
}

# Layout and UI
row0_0, row0_1, row0_2 = st.columns((1, 12, 1))
with row0_1:
    # Display header with logo
    st.markdown(
        f"""
        <div style="text-align: center; padding-bottom:0px; margin-top:-50px;">  
            <img src="{logo_path}" width="{logo_width}">
        </div>
        <div style="text-align: center; margin-top:-10px">  
            <h1>Generate Audiences and Images</h1> 
            <h4>Sends sample data to Gemini so that it can automatically create audiences</h4>
            </br>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.container(border=True):
    row0_0, row0_1, row0_2 = st.columns((.1, 40, .1))
    row1_0, row1_1, row1_2 = st.columns((1, 12, 1))
    row02_0, row2_1, row2_2 = st.columns((1, 12, 1))
    with row0_1:
        # Display sample data
        st.write("Sample Data")
        st.dataframe(df, hide_index=True, use_container_width=True)

        # Example recommendation output
        recoexample = """
        Segment: {segment name based on customer data and company insight}
        Customers: {Emails of customers that match segment}
        Most likely to respond to: {marketing channel(s)}
        Language: {languages they speak}
        Product: {pull product info from data}
        Current Weather: {current weather(s)}
        """

    with row2_1:
        # Text area for company background
        recotask = st.text_area("Company Background", f"{company}")
        # Text area for recommendation task
        recotask += st.text_area(
            "Recommendation Task",
            """Create 3 Audience Segments of users to place each user into for the company. 
            Then return the Audience with each user identified in them and get me a list of who is most likely to respond to an advertisement on which marketing channel. 
            What language should be in the ad."""
        )

        # Construct the prompt for Gemini
        recoprompt = f"""Your Task: {recotask}. Use all the following data examples: {df} 
        Use bulletpoints to separate ideas and be verbose in your responses. 
        Be sure one of the bulletpoints is 'current weather' When generating a recommendation for what images might be shown to a customer be sure to consider 
        the segment name, weather, location, make all images of the product indoors. 
        Here is an example of a good output {recoexample}"""

        # Button to trigger the generation process
        go_button_sql = st.button("Generate")

# --- Process the request ---
if go_button_sql:
    row4_0, row4_1, row4_2 = st.columns((1, 12, 1))
    with st.spinner(text="Generating responses..."):
        # Remove existing images in the data folder
        for filename in os.listdir("./data"):
            if filename.endswith(f".jpg"):
                filepath = os.path.join("./data", filename)
                os.remove(filepath)
                print(f"Image removed: {filepath}")

        with row4_1:
            # Get the first text response from Gemini
            textresponse = vertex_text(recoprompt)

            st.session_state.output0 = textresponse
            st.markdown(st.session_state.output0)

        with row4_1:
            # Define example JSON structure for image generation recommendations
            examplejson = {
                "type": "object",
                "properties": {
                    "Audiences": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["Audience_Name", "image_description"],
                            "properties": {
                                "Audience_Name": {
                                    "type": "string",
                                    "description": "The name of the audience which is being targeted"
                                },
                                "image_description": {
                                    "type": "string",
                                    "description": "The description of the image to be generated."
                                }
                            }
                        }
                    }
                }
            }

            # Construct the prompt for generating JSON output
            jsonprompt = f"""
            Your Task:
            Generate a JSON output of image generation recommendations.
            Be sure to create valid JSON.
            Return only raw JSON.
            No children in the images, so change children to adults and change families to adults.
            Put each background in a different setting.
            Be sure to generate recommendations for every audience return a minimum of 3 audiences returned.
            Here is some example JSON of an end result be strict in your adherence to this sample: {examplejson}
            Use the following generated text for inspiration on locations to place the product image: {textresponse}.  
            Only generate 3 image descriptions total.
            adhere to the sample JSON for the output, no markdown. 
            for audience names do not include any forward slashes since that can break lookups on the os level.
            don't include any text overlay on the images.
            """

            # Generate JSON from Gemini
            json_result = vertex_text_strict(jsonprompt, examplejson)
            print(json_result)
            st.title("JSON of image prompts, made more creative by Gemini")
            st.json(json_result)

            # Convert JSON result to a Python dictionary
            json_data = json.loads(json_result)
            print("After conversion to dict")
            print(json_data)

            # Display example images for each audience segment
            st.title('Example Images for Audience Segment')
            audiences = json_data["Audiences"]
            print(audiences)
            for audience in audiences:
                audience_name = audience["Audience_Name"]
                image_description = audience["image_description"]
                image_prompt = f"{image_description}"
                neg_prompt = ""
                photo_features="as if shot on a DSLR"

                st.title(f"{audience_name}")
                output3=f"Image Description:{image_description}"
                st.markdown(output3)

                imgfilename = f"{audience_name}-gen-personalization"

                images=vertex_image(image_prompt,photo_features,imgfilename) #generate images
                display_images(images,imgfilename) #display the images (loop through however many were returned)
                
                with row4_1:
                    print(f"entering image loop: {imgfilename}")
                    i = 0
                    while i < 5:
                        try:
                            filepath = f'./data/{i}-{imgfilename}.jpg'
                            if os.path.exists(filepath):
                                st.image(f"./data/{i}-{imgfilename}.jpg", width=400)
                            else:
                                print("image does not exist")
                        except Exception:
                            print('done-loading-image-creation')
                            pass
                        i+=1