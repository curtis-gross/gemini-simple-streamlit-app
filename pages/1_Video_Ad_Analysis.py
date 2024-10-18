#Copyright 2024 Google LLC
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#    https://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
import streamlit as st #pip install streamlit
import vertexai # pip install vertexai
import io
import os
import google.auth
from google.auth import transport
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
import pandas as pd
import time
import plotly.express as px
import json
import re
from datetime import timedelta
from central_config import keyload
credentials, project_id = keyload()
from central_config import styling, region, display_additional_notes, load_config_file, gs_folder, bucket_name
from central_config import genai_video_json_strict, time_to_seconds, load_video, get_storage_client

st.set_page_config(layout="wide")

##################################################
# A more advanced and yet messy example of how to take an Video and convert it into a response in a Json Schema
# You need a video which will need to be in a GCP bucket defined in the 'central_config.py' file
# then you can tweak the data/config.json file to look for the right keywords in that bucket for the right video file.
# in order to use Gemini to produce structured json schema return and visualize it.
##################################################


if 'json_results_ads' not in st.session_state:
    st.session_state.json_results_ads = {}  # Initialize as an empty dictionary

config = load_config_file("data/config.json")
if config:
    ad_keywords = config["ad_keywords"]
    demo_mode = config["demo_mode"]
    sleep_time = config["sleep_time"]
    logo_path = config["logo_path"]
    logo_width = config["logo_width"]
    company = config["company"]
    brand_names = config["brand_names"]
    region = config["region"]
    imagen_version = config["imagen_version"]
    model = config["model"]
    customer_tags = config["customer_tags"]
    p41 = config["p41"]
    p42 = config["p42"]
    p43 = config["p43"]
    contentUrls = config["contentUrls"]
    focus_products = config["focus_products"]
else:
    st.write("error loading config file")

if demo_mode == False:
    st.write(f"Demo Mode is currently off, read notes in code.")
    #configure on data/config.json
    #demo mode is basically 'caching' the results.  You want this to reduce demo costs.  
    # Run the analyize, hit save, do this for all videos, then turn on demo mode.

st.markdown(f"""{styling}""",unsafe_allow_html=True)

PROJECT_ID = project_id  # @param {type:"string"}
REGION = region # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location=REGION)

row0_0, row0_1, row0_2 = st.columns((1,12,1))
with row0_1:
    st.markdown(
        f"""
        <div style="text-align: center;padding-bottom:0px;margin-top:-50px;">  
            <img src="{logo_path}" width="{logo_width}">
        </div>
        <div style="text-align: center;margin-top:-10px">  
            <h1>Video Analysis w/Vertex AI</h1> 
            </br>
        </div>
        """,
        unsafe_allow_html=True
    )

#user interface

row1_0, row1_1, row1_2, row1_3 = st.columns(vertexai(1,12,12,1))
with row1_1:
    storage_client = get_storage_client()
    try:
        blobs = storage_client.list_blobs(bucket_name)

        video_file_names = [
            blob.name
            for blob in blobs
            if blob.name.endswith((".mp4", ".avi", ".mov", ".webm")) and
            all(word in blob.name.lower() for word in ad_keywords)
        ]
    except google.cloud.exceptions.NotFound:
        st.error(f"Error: Bucket '{bucket_name}' not found.")
        video_file_names = []  # Set to empty list to avoid further errors
    
   
    video_select = st.selectbox(
        "Video to Analyze",
        video_file_names,  # Pass the list of video file names directly
        index=0,  # Set default selection
        key="video_select_box"  # Important: Unique key for the selectbox
    )

    selected_video_url = f"https://storage.googleapis.com/{gs_folder}{video_select}"
    vertex_selected_video = f"gs://{gs_folder}{video_select}"
    print(f"vid selected {selected_video_url}")
    # Trigger rerun when the video selection changes
    if st.session_state.get("prev_selected_video_url") != video_select:
        st.session_state["prev_selected_video_url"] = video_select
        st.rerun()

    main_prompt = st.text_area("prompt",
f"""Prompt:
You are a meticulous brand compliance manager tasked with identifying and analyzing brand visibility within a video. Scrutinize this video with a keen eye, focusing on the appearance of brands and their screen time.
Tasks:

Brand Identification:
Identify all unique brands that appear in the video.
Note the first time that each brand appears on screen.

Time on Screen Tracking:
Create a table named TimeOnScreen with the following columns:
Brand: The name of each brand identified in the video.
TimeOnScreenStart: The time in seconds when a brand first appears on screen.
TimeOnScreenEnd: The time in seconds when the brand is no longer visible on screen.
This is Important!!! be sure to capture all moments that the brand appears in the video.

Important:
If a brand appears multiple times throughout the video (in separate instances), create a new row in the table for each appearance with the corresponding time in seconds.
Percentage of Video Time:
For each brand, calculate the percentage of the total video length during which the brand is visible on screen.

Additional Notes: (use markdown for this, everything in a bullet list)
Give an overview of what the ad is about.
What improvements could be made to the ad?
Who is the ad for?
What age range is the ad for?
What is the overall style of the ad?
What is the setting for the ad?
What is the music style of the ad?
What is the color pallete of the ad?
Who are the main characters?
Is the story front-loaded so as to reduce someone skipping the video part way through?
Are there familiar looking faces in the video that the target market can connect with? 
How long does it take for a person to be on screen?
Does the music match the video from a pacing and style standpoint?
Does the creative style match the target audience?
Is the brand integrated into the video in a natural way in the first 5 seconds?
Is the brand name mentioned in the audio of the video?
Does the video have quick cuts between scenes, what is the average time between a cut?
Does the video feature humor or suspense?  Rank them on a scale of 1-10 with 10 being the highest.
What is the main message or idea the advertisement is trying to convey?
Who is the target audience for this advertisement?
What emotions does the advertisement evoke? How does it make you feel?
What are the primary visual elements used in the advertisement?
Does the advertisement effectively grab your attention? Why or why not?
Are there any specific symbols or imagery used in the advertisement? What do they represent?
What kind of language and tone of voice is used in the advertisement?
Is the message of the advertisement clear and easy to understand?
How does the advertisement differentiate itself from competitors' ads?
Does the advertisement offer any specific call to action?
What values or lifestyle does the advertisement promote?
Do you think the advertisement is ethical and responsible? Why or why not?
Does the advertisement create a memorable impression? Why or why not?
Is the advertisement persuasive? Does it make you want to buy the product or service?
What cultural references or stereotypes are present in the advertisement?
Does the advertisement utilize any humor or shock value? Is it effective?
Is the advertisement appropriate for all audiences? Why or why not?
What are the potential consequences of this advertisement, both positive and negative?
If you could change one thing about this advertisement, what would it be?
Overall, would you consider this advertisement to be successful? Why or why not?

Precision is Paramount: Ensure that timestamps are as accurate as possible.
Clarity: Organize your data in a clear, easy-to-read format

If any items in the data are missing make them None, unless it is VideoTime then it must be a number. 
Do not skip any of the main elements. 
If a brand value is none, you can skip to the next main element.
Please limit the total number of TimeOnScreenTracking to 20 total.
""")
   #main_prompt += f"Here is an example of a good output: {example_analysis}"

    json_schema = {
    "type": "object",
    "properties": {
        "BrandComplianceAnalysis": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
            "Brand": {
                "type": "string",
                "description": "The name of the brand being analyzed."
            },
            "Analysis": {
                "type": "string",
                "description": "High-level analysis of the advertisement for this brand."
            }
            },
            "required": ["Brand", "Analysis"]
        }
        },
        "BrandIdentification": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
            "BrandName": {
                "type": "string",
                "description": "The name of the brand."
            },
            "BrandAppearanceTime": {
                "type": "integer",
                "description": "The time (in seconds) when the brand first appears in the video."
            }
            },
            "required": ["BrandName", "BrandAppearanceTime"]
        }
        },
        "TimeOnScreenTracking": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
            "Brand": {
                "type": "string",
                "description": "The name of the brand."
            },
            "TimeOnScreenStart": {
                "type": "integer",
                "description": "The time (in seconds) when the brand appears on screen."
            },
            "TimeOnScreenEnd": {
                "type": "integer",
                "description": "The time (in seconds) when the brand disappears from the screen."
            }
            },
            "required": ["Brand", "TimeOnScreenStart", "TimeOnScreenEnd"]
        }
        },
        "PercentageOfVideoTime": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
            "Brand": {
                "type": "string",
                "description": "The name of the brand."
            },
            "TotalTimeOnScreen": {
                "type": "integer",
                "description": "The total time (in seconds) the brand is on screen."
            },
            "VideoTime": {
                "type": "integer",
                "description": "The total duration of the video (in seconds)."
            },
            "Percentage": {
                "type": "string",
                "description": "The percentage of the video time that the brand is on screen."
            }
            },
            "required": ["Brand", "TotalTimeOnScreen", "VideoTime", "Percentage"]
        }
        },
        "AdditionalNotes": {
        "type": "string",
        "description": "Additional feedback and answers to questions in markdown format with bullet points."
        }
    }
    }

    go_button = st.button("Analyze Video")
with row1_2:
    video_placeholder = load_video(selected_video_url)

video_file = Part.from_uri(
    mime_type="video/mp4",
    uri=f"gs://{gs_folder}{video_select}")
print(video_file)


def fix_common_json_errors(json_string):
    # Replace single quotes with double quotes (common error)
    json_string = json_string.replace("'", '"')
    
    # Try to fix missing commas (this can be tricky, so use with caution)
    # Example: Find missing commas before closing curly braces in dictionaries
    json_string = json_string.replace("}", "},")
    # ... (Add more pattern-based fixes as needed)
    print("fixed")
    return json_string


# Demo mode save button
if demo_mode == False:
    with st.form("Save Form"):
        st.write("Like these results? Save them and then enable demo mode.  Note that all the videos must have saved results for demo mode to work best")
        submitted = st.form_submit_button("Save Data")
        if submitted:
            print("####################Save Button activated")
            # Get the data from your function
            data_to_save = st.session_state.json_results_ads

            # Create the 'data' folder if it doesn't exist
            if not os.path.exists("data"):
                os.makedirs("data")

            # Save the data to a JSON file
            with open(f"data/{video_select}.json", "w") as f:
                json.dump(data_to_save, f, indent=4)  # 'indent' makes the JSON more readable

            st.write(f"Data saved to data/{video_select}.json!")

if go_button:
    with st.spinner(text="Analyzing Video..."):
        with st.container(border=False):
            if demo_mode == False:
                ##################send video and prompt data to gemini to analyze the video
                video_analysis = genai_video_json_strict(video_file, main_prompt, json_schema)
                json_results = json.loads(video_analysis)
                st.session_state.json_results_ads = json_results
                print(f"###############Json Results Loaded################## \n {json_results}")

################ Static results for consistency####################################################################################################################################
            else:
                time.sleep(sleep_time)
                with open(f"data/{video_select}.json", "r") as f:
                    loaded_data = json.load(f)
                    print("Loaded file from filesystem")

                json_results = loaded_data
                print(json_results)


########################################    
            row2_0, row2_1, row2_2, = st.columns((1,9,1))
            with row2_1:
                # Extract data correctly (handling potential errors)
                brand_compliance_analysis = json_results.get("BrandComplianceAnalysis", [])
                # Check if brand_compliance_analysis is a list, and get the first element (the dictionary you need)
                if isinstance(brand_compliance_analysis, list) and brand_compliance_analysis:
                    analysis_data = brand_compliance_analysis[0] 
                    video_description = analysis_data.get("Analysis")
                else:
                    video_description = None
                   

            
                st.title(f"Brand Compliance Analysis: {company} Advertisement") 
                st.markdown(f"{video_description}")

# ############################################## TIMESTAMPS -  take brand table, pass it to gemini and ask ""given the following data, grab 3 unique rows with different brands""

             
            cols_per_row = 3
            current_row = 3  # Starting row index 

            # Initialize row counters
            row_index = 0
            col_index = 0
            
            video_brand = json_results.get("BrandIdentification", [])
            time_on_screen_data = json_results.get("TimeOnScreenTracking", [])
            table_data = time_on_screen_data
        if table_data:
            with row2_1:
                st.title(f"{company} Moments within Advertisement")
                while row_index < len(table_data):
                    cols = st.columns(cols_per_row)  # Create the columns for this row

                    for i in range(cols_per_row):
                        if row_index < len(table_data):
                            item = table_data[row_index]
                            with cols[i]:  # Use the column object directly
                                with st.container(border=True):
                                    stmp = item['TimeOnScreenStart']
                                    print(stmp) #prints {'00:00'}
                                    #stmp = time_to_seconds(str(stmp))
                                    brand = item['Brand']
                                    st.markdown(
                                    f"<h4 style='min-height:100px'>{brand} Moment</h4>",
                                    unsafe_allow_html=True,
                                    )

                                    st.markdown(
                                        f'<video controls muted style="max-width:100%;padding:5px;">  <source src="{selected_video_url}#t={stmp}" type="video/mp4"></video><br>',
                                        unsafe_allow_html=True,
                                    )
                                    st.write(f"**TimeStamp:** {item['TimeOnScreenStart']}")
                            row_index += 1
                        else:
                            break  # Stop if we've run out of items
            col_index = 0
        else:
                print("missing data")
                # Move to the next row if we've filled the current one
                

            
######################################            
        with row2_1:
            st.title(f"{company} Appears within first 4 seconds?") 

            brand_identification_data = json_results.get("BrandIdentification", [])
            print(f"brand_identification_data: {brand_identification_data}")
            # Check if brand_identification_data is a list
            if isinstance(brand_identification_data, list):
            # Iterate through the brands and display their information
                for brand in brand_identification_data:
                    st.markdown(f"**{brand['BrandName']}**: {brand['BrandAppearanceTime']} seconds")
                    brand_time = int(float(brand['BrandAppearanceTime']))
                    if brand_time < 4:
                        st.markdown("- **Yes**")
                    else:
                        st.markdown("- **No**")
               
                
# ######################################
        with row2_1:
            st.title("Brand total time on screen") 
            # Extract data correctly (handling potential errors)
            percent_on_screen = json_results.get("PercentageOfVideoTime", [])
            print(f"percent_on_screen :{percent_on_screen}")
           
            if isinstance(percent_on_screen, list):
            # Iterate through the brands and display their information
                for brand in percent_on_screen:
                    st.markdown(f"**{brand['Brand']}**: total time on screen: {brand['TotalTimeOnScreen']}")
                    brandname = brand['Brand']
                    print(f"Brand Name: {brandname}, company : {company}")
                    brand_time = int(float(brand['TotalTimeOnScreen']))
                    brand_names_lower = [name.lower() for name in brand_names]
                    if brandname.lower() in brand_names_lower:   
                        print("Brand name matches data")
                        perc = brand['Percentage']
                        if type(perc) == str:
                            brand_percentage = brand['Percentage'].strip('%')
                            brand_percentage = float(brand_percentage)
                        elif isinstance(perc, (int, float)):  # Handle both ints and floats
                            brand_percentage = float(perc)     # Ensure it's a float if it was originally an int
    
                        print(f"percent of video that is {company}: {brand_percentage}")
                
            # Extract and transform data for the pie chart
            other_brands_percentage = 100 - brand_percentage  # Calculate the percentage for other brands
            pie_chart_data = {
                'Brand': [f'{company}', 'Other'],
                'Percentage': [brand_percentage, other_brands_percentage]
            }

            df_percentage = pd.DataFrame(pie_chart_data)

            # Create the Plotly pie chart
            fig = px.pie(
                df_percentage,
                values='Percentage',
                names='Brand',
                title=f'{company} vs. Total Time Distribution (%)',
                hole=0.3  # Optional: donut chart style
            )

            # Customize layout (optional)
            fig.update_traces(textposition='inside', textinfo='percent+label', insidetextfont=dict(color='white'))

            # Display the chart in Streamlit
            st.plotly_chart(fig)


# ######################################       
    
        with row2_1:
            print(json_results)
            if json_results:  # Ensure json_results is not empty
                st.title("Advertisement Metadata")
                propname = "AdditionalNotes"
                display_additional_notes(json_results, propname)

