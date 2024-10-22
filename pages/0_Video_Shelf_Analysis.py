#these work with appengine and ubuntu 22 / python 3.12
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
import google.auth
from google.auth import transport
from google.cloud import storage
from streamlit.runtime.scriptrunner import get_script_run_ctx #for multi-threading in streamlit
from streamlit.runtime.scriptrunner import add_script_run_ctx #for multi-threading in streamlit
from vertexai.language_models import TextGenerationModel
from vertexai.preview.vision_models import Image, ImageGenerationModel
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
import tempfile
import pandas as pd
import time
import random
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import json
import os
from google.cloud import bigquery
client = bigquery.Client()
from central_config import keyload
credentials, project_id = keyload()
from central_config import styling, region, load_config_file, gs_folder, bucket_name
from central_config import genai_video_json_strict, time_to_seconds, load_video, get_storage_client

st.set_page_config(layout="wide")

##################################################
# A more advanced and yet messy example of how to take an Video and convert it into a response in a Json Schema
# You need a video which will need to be in a GCP bucket defined in the 'central_config.py' file
# then you can tweak the data/config.json file to look for the right keywords in that bucket for the right video file.
# in order to use Gemini to produce structured json schema return and visualize it.
##################################################

config = load_config_file("data/config.json")
if config:
    shelf_keywords = config["shelf_keywords"]
    demo_mode = config["demo_mode"]
    sleep_time = config["sleep_time"]
    logo_path = config["logo_path"]
    logo_width = config["logo_width"]
    company = config["company"]
    #brand_names = config["brand_names"]
    region = config["region"]
    imagen_version = config["imagen_version"]
    model = config["model"]
    customer_tags = config["customer_tags"]
    p41 = config["p41"]
    p42 = config["p42"]
    p43 = config["p43"]
    contentUrls = config["contentUrls"]
    focus_products = config["brand_names"]
else:
    st.write("error loading config file")

if demo_mode == False:
    st.write(f"Demo Mode is currently off, read notes in code.")
    #configure on data/config.json
    #demo mode is basically 'caching' the results.  You want this to reduce demo costs.  
    # Run the analyize, hit save, do this for all videos, then turn on demo mode.

if 'json_results_shelf' not in st.session_state:
    st.session_state.json_results_shelf = {}  # Initialize as an empty dictionary

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

row1_0, row1_1, row1_2, row1_3 = st.columns((1,12,12,1))
with row1_1:
    storage_client = get_storage_client()
    try:
        blobs = storage_client.list_blobs(bucket_name)

        video_file_names = [
            blob.name
            for blob in blobs
            if blob.name.endswith((".mp4", ".avi", ".mov", ".webm")) and
            all(word in blob.name.lower() for word in shelf_keywords)
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

    json_schema = {
  "type": "object",
  "properties": {
    "analysis": {
      "type": "object",
      "properties": {
        "videoDescription": {
          "type": "string",
          "description": "High-level analysis of the video. What store or place it was filmed, and overall impression of the scene."
        },
          "datetime": {
          "type": "number",  
          "description": "a datetime timestamp in the format of the current epoch time for when the json was created"
        },
        "noStock": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "Category": {
                "type": "string",
                "description": "Category of the out-of-stock product (e.g., home goods, toys, candy)."
              },
              "Product": {
                "type": "string",
                "description": "Estimated name of the out-of-stock product."
              },
              "Need": {
                "type": "string",
                "description": "Urgency of restocking (High, Medium, Low) based on shelf emptiness. Do your best to find at least 1 item that is low stock"
              },
              "startTimestamp": {
                "type": "string",
                "description": "Timestamp when the out-of-stock situation is observed (e.g., '00:08') if none, just enter 00:00."
              }
            },
            "required": ["Category", "Product", "Need", "startTimestamp"]
          }
        },
        "productTable": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "Category": {
                "type": "string",
                "description": "Category of the product (e.g., home goods, toys, candy)."
              },
              "Product": {
                "type": "string",
                "description": "Name of the product."
              },
              "Count": {
                "type": "integer",
                "description": "Estimated count of the product on the shelf."
              },
              "startTimestamp": {
                "type": "string",
                "description": "Timestamp when the product is first seen (e.g., '00:08') if none, just enter 00:00."
              },
              "Price": {
                "type": "string",
                "description": "Price of the product, pulled from the tags on the shelves. "
              }
            },
            "required": ["Category", "Product", "Count", "startTimestamp", "Price"]
          }
        },
        "totals": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "Brand": {
                "type": "string",
                "description": "Name of the brand."
              },
              "TotalCount": {
                "type": "integer",
                "description": "Total count of products for this brand."
              }
            },
            "required": ["Brand", "TotalCount"]
          }
        },
        "shelfSpace": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "Brand": {
                "type": "string",
                "description": "Name of the brand."
              },
              "TotalTimeOnScreen": {
                "type": "integer",
                "description": "Total time (in seconds) the brand is visible on screen."
              },
              "VideoTime": {
                "type": "integer",
                "description": "Total duration of the video in seconds."
              },
              "Percentage": {
                "type": "string",
                "description": "Percentage of video time the brand occupies."
              }
            },
            "required": ["Brand", "TotalTimeOnScreen", "VideoTime", "Percentage"]
          }
        },
        "audio": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "Timestamp": {
                "type": "string",
                "description": "Timestamp when the audio is heard (e.g., '00:08') if none, just enter 00:00."
              },
              "Audio": {
                "type": "string",
                "description": "Description of the audio heard."
              }
            },
            "required": ["Timestamp", "Audio"]
          }
        },
        "optimizationRecommendations": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "recommendation": {
                "type": "string",
                "description": "Recommendation for improving shelving. Always produce at least 3 suggestions."
              },
              "description": {
                "type": "string",
                "description": "Detailed description of the recommendation."
              }
            },
            "required": ["recommendation", "description"]
          }
        },
        "shelfAccessibility": {
          "type": "object",
          "properties": {
            "lowestShelfAccessibility": {
              "type": "string",
              "description": "Description of who can access the lowest shelf and what products are on it."
            },
            "text": {
              "type": "string",
              "description": "Additional text related to shelf accessibility."
            }
          },
          "required": ["lowestShelfAccessibility", "text"]
        },
        "furtherConsiderations": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "consideration": {
                "type": "string",
                "description": "Additional consideration or observation about the video."
              },
              "description": {
                "type": "string",
                "description": "Detailed description of the consideration."
              }
            },
            "required": ["consideration", "description"]
          }
        }
      },
      "required": [
        "videoDescription",
        "datetime",
        "noStock",
        "productTable",
        "totals",
        "shelfSpace",
        "audio",
        "optimizationRecommendations",
        "shelfAccessibility",
        "furtherConsiderations"
      ]
    }
  },
  "required": ["analysis"]
}

    import datetime

    # Get the current datetime
    now = datetime.datetime.now()

    # Get the timestamp as a numeric value (seconds since epoch)
    numeric_timestamp = now.timestamp()

    print(numeric_timestamp)

    main_prompt = st.text_area("prompt",
    f"""You are a store manager, Provide an analysis of this video, be verbose with your reasoning. Do your best and always produce results. 
    Always produce an output, do your best.
    audio: if none, just enter 00:00.
    datetime: For the datetime field, populate it with the current datetime in numeric values which is: {numeric_timestamp}
    Be as precise as possible with your count. It looks like packages are stocked a few packages deep in each shelf, you are allowed to estimate total values based on that knowledge.
    Here are the focus products: Spend extra time looking for the following products if they exist in the video: {focus_products}
    Try to identify at least 15 products or sub-products, no more than 35 though.
    Try to identify at least 5 item prices. 
    Price: Note that some tags might have an offer similar to 2 / $5 or 2 for 5. This is a division problem, 5 divided by 2, so 2.50.
    Price: You should get a price per unit for all prices, example $2.50 or $.89 . 
    Price: should be stored as a string.
    the final output will need to be stored within a json structure
    Make more considerations in 'furtherConsiderations' along the following items:
    How organized do the shelves look?
    How many different items do you see?
    What would be some tips to organize better?
    Is there any color coordination that could take place, or is taking place?
    What could be done to optimize the layout?
    Would you consider the display appealing to the eye?
    Who is the target buyer for this?
    Provide a markdown bullet list of the positive as well as the negative observations and a summary.
    when returning empty values, use 'None' rather than 'Null'
     """)
    go_button = st.button("Analyze Video")
    
    # Demo mode save button

    
    if demo_mode == False:  
      saved = st.button("Save Results Locally")
      if saved:
          print("#####################Save Button activated")
          # Get the data from your function
          if st.session_state.json_results_shelf and any(st.session_state.json_results_shelf.values()): 
              data_to_save = st.session_state.json_results_shelf
              # Create the 'data' folder if it doesn't exist
              if not os.path.exists("data"):
                  os.makedirs("data")

              # Save the data to a JSON file
              with open(f"data/{video_select}.json", "w") as f:
                  json.dump(data_to_save, f, indent=4)  # 'indent' makes the JSON more readable

              st.write(f"Data saved to data/{video_select}.json!")
          else:
              st.write(f"You need to run the Analyize option first")
              pass
    
    #bqsave = st.button("Save Results to BQ")  ############################################################################uncomment this line to show a button to save to bq
    bqsave = False  #comment this out too
    print(f"data/{video_select}.json")
    print(dir(bigquery.SourceFormat))
    if bqsave:
        print("#####################Save to BQ Button activated")
        with open(f"data/{video_select}.json", "r") as f:
            st.session_state.json_results_bq = json.load(f)
            print("Loaded file from filesystem")
        data_to_save = st.session_state.json_results_bq
        data_to_save = [data_to_save]  # Add brackets to create a list
        # Use SourceFormat.JSON for a single JSON object
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,  # Changed source format
            autodetect=True,
        )
        table_id = "bb-test-404418.pepsico.videoanalysis"
        load_job = client.load_table_from_json(
            data_to_save, table_id, job_config=job_config
        )
        load_job.result()

        if load_job.errors:
            st.error(f"Error loading data to BigQuery: {load_job.errors}")
        else:
            st.success("Data loaded to BigQuery successfully!")


with row1_2:
    video_placeholder = load_video(selected_video_url)

# Define custom colors for each brand in plotly(adjust as needed)
colors = {
    'Dr. Pepper':'#A1522D',
    'Mtn Dew':'#82B366',
    'Cherry Coke':'#C62828',
    'Coke': '#C62828',
    'Coca Cola': '#C62828',  # Deep Red
    'Coca-Cola': '#C62828',  # Deep Red
    'Coca-Cola Zero': '#3E2723',  # Dark Brown (almost black)
    'Diet Coke': '#C0C0C0',  # Light Gray (less bright than silver)
    'Diet Pepsi': '#C0C0C0',  # Light Gray (less bright than silver)
    'Coca-Cola Spiced': '#795548',  # Muted Brown
    'Fanta': '#FB8C00',  # Burnt Orange
    'Sprite': '#81C784',  # Pale Green
    'Starry': '#81C784',  # Pale Green
    'Other Brands': '#64B5F6',  # Light Blue
    'Pepsi': '#64B5F6',  # Light Blue
}

video_file = Part.from_uri(
    mime_type="video/mp4",
    uri=f"gs://{gs_folder}{video_select}")
print(video_file)


if go_button:
    with st.spinner(text="Analyzing Video..."):
        with st.container(border=False):
            #################send video and prompt data to gemini to analyze the video
            if demo_mode == False:
                video_analysis = genai_video_json_strict(video_file, main_prompt, json_schema)
                json_results = json.loads(video_analysis)
                st.session_state.json_results_shelf = json_results
                print(f"###############Json Results Loaded################## \n {json_results}")
            #Static results for consistency####################################################################################################################################
            else:
                time.sleep(sleep_time)
                with open(f"data/{video_select}.json", "r") as f:
                    loaded_data = json.load(f)
                    print("Loaded file from filesystem")

                json_results = loaded_data
                st.session_state.json_results_shelf = json_results
                print(json_results)
#########################################
            
            row2_0, row2_1, row2_2, = st.columns((1,9,1))
            with row2_1: 
                video_description = json_results["analysis"]["videoDescription"]

########################################                
                st.title("Analysis") 
                st.markdown(f"{video_description}")
######################################            
                st.title("Recomendation") 

                for recommendation in json_results["analysis"]["optimizationRecommendations"]:
                    st.markdown(f"- **{recommendation['recommendation']}**: {recommendation['description']}")

###################################### Out of stock
                #no_stock_data = json_results['analysis']['noStock']

                cols_per_row = 3
                row_index = 0

                st.title("Out-of-Stock Products in Video")
                while row_index < len(json_results['analysis']['noStock']):
                    cols = st.columns(cols_per_row)
                    for i in range(cols_per_row):
                        if row_index < len(json_results['analysis']['noStock']):
                            item = json_results['analysis']['noStock'][row_index]
                            with cols[i]:
                                with st.container(border=True):
                                    # Extract timestamp and handle missing values (if any)
                                    timestamp = item.get('startTimestamp', "00:00")  # Default to "00:00" if timestamp is missing
                                    timestamp_seconds = time_to_seconds(timestamp)  # Convert to seconds

                                    # Display product information
                                    st.markdown(f"### {item['Product']}")
                                    st.write(f"**Need:** {item['Need']}")
                                    st.write(f"**Start Time:** {timestamp}")

                                    # Embed video snippet starting at the specified timestamp
                                    st.markdown(
                                        f'<video width="240" controls muted style="max-width:100%;padding:5px;"> <source src="{selected_video_url}#t={timestamp_seconds}" type="video/mp4"></video>',
                                        unsafe_allow_html=True,
                                    )

                            row_index += 1
                        else:
                            break

######################################
            
                st.title("In Stock Product Analysis")
                # Define data as a list of dictionaries
                brand_table_data = json_results["analysis"]["productTable"]
                brandtable = brand_table_data
                df_brandtable = pd.DataFrame(brand_table_data)

                # Calculate average price for each brand (handling potential errors)
                def calculate_avg_price(prices_str):
                    try:
                        if prices_str == 'None':  # Handle None values
                            return 0
                        prices_list = [float(p.replace("$", "")) for p in prices_str.split(", ")]
                        return sum(prices_list) / len(prices_list)
                    except ValueError:
                        st.error(f"Invalid price format: '{prices_str}'")  # Log error in Streamlit
                        return None

                df_brandtable["AvgPrice"] = df_brandtable["Price"].astype(str).apply(calculate_avg_price)

                # Filter out rows with missing prices to avoid errors in the chart
                df_brandtable_filtered = df_brandtable.dropna(subset=["AvgPrice"])

                # Create Plotly bubble chart (with filtered data)
                fig = px.scatter(
                    df_brandtable_filtered, 
                    x="Product", 
                    y="AvgPrice", 
                    size="Count", 
                    color="Product", 
                    hover_name="Product",
                    color_discrete_map=colors, 
                    size_max=60
                )
                fig.update_traces(marker=dict(sizemode='area'))

                # Add title and labels
                fig.update_layout(
                    title='Product Count and Average Price by Brand',
                    xaxis_title='Product',
                    yaxis_title='Average Price ($)'
                )

                # Display the chart in Streamlit
                st.plotly_chart(fig)

                
##############################################   BRAND % SHELF SPACE CHART             

        row2_0, row2_1, row2_2, row2_3 = st.columns((1,4,4,1))
        product_table = json_results["analysis"]["productTable"]
        with row2_1:
            # Convert product data to DataFrame
            df_products = pd.DataFrame(product_table)

            # Bar chart of product counts
            fig_bar = px.bar(
                df_products,
                x="Product",
                y="Count",  # Changed from 'TotalCount' to 'Count'
                title="In Stock Count",
                text_auto=True,  # Show count values on bars
                labels={"Count": "Total Count"},
            )
            fig_bar.update_layout(showlegend=False)  # No need for a legend with one color
            st.plotly_chart(fig_bar)

        with row2_2:
            # Optionally: Create a pie chart of product proportions
            fig_pie = px.pie(
                df_products,
                values='Count',
                names='Product',
                title='Products In Stock',
                hole=0.3  # Donut style
            )
            st.plotly_chart(fig_pie)

                

############################################## TIMESTAMPS -  take brand table, pass it to gemini and ask ""given the following data, grab 3 unique rows with different brands""
        def time_to_seconds(time_str):
            """Converts a time string in MM:SS format to total seconds."""
            minutes, seconds = map(int, time_str.split(':'))
            return minutes * 60 + seconds

        
        cols_per_row = 3
        current_row = 3  # Starting row index 

        # Initialize row counters
        row_index = 0
        col_index = 0
        row4_0, row4_1, row4_2, = st.columns((1,9,1))
        brand_table_data = brandtable
        with row4_1: 
            st.title("Key Products Identified in Video")
            while row_index < len(brand_table_data):
                cols = st.columns(cols_per_row)  # Create the columns for this row
                for i in range(cols_per_row):
                    if row_index < len(brand_table_data):
                        item = brand_table_data[row_index]
                        with cols[i]:  # Use the column object directly
                            with st.container(border=True):
                                                            
                                stmp = item['startTimestamp']
                                print(f"STMP Values: {stmp}") #prints {'00:00'}
                                if stmp is None:
                                    stmp = 0
                                else:
                                    stmp = time_to_seconds(str(stmp))
                                st.markdown(f"### {item['Product']}")
                                #st.write(f"**Product:** {item['Product']}")
                                st.write(f"**Count:** {item['Count']}")
                                if 'Price' in item and item['Price'] is not None:
                                    cost = item['Price'].replace("$", "")
                                else:
                                    cost = 'N/A'  # Or any other default value you prefer
                                st.write(f"**Price:** ${cost}")
                                st.write(f"**Start Time:** {item['startTimestamp']}")
                                st.markdown(
                                    f'<video controls muted style="max-width:100%;padding:5px;">  <source src="{selected_video_url}#t={stmp}" type="video/mp4"></video><br>',
                                    unsafe_allow_html=True,
                                )
                                
                        row_index += 1
                    else:
                        break  # Stop if we've run out of items

                # Move to the next row if we've filled the current one
                col_index = 0


######################################               
        row4_0, row4_1, row4_2 = st.columns((1,9,1))
        with row4_1:
            st.title("Further Considerations")
            for consideration in json_results["analysis"]["furtherConsiderations"]:
                st.markdown(f"- **{consideration['consideration']}**: {consideration['description']}")

    ######################################  
            # Extract shelf accessibility data
            shelf_accessibility = json_results["analysis"].get("shelfAccessibility", {}) # Use get() for safety

            # Display shelf accessibility information
            st.title("Shelf Accessibility")

            for key, value in shelf_accessibility.items():
                if key == 'totalShelves':
                    st.markdown(f"- **Total Shelves:** {value}")
                elif key == 'lowestShelfAccessibility':
                    st.markdown(f"- **Lowest Shelf Accessibility:** {value}")

