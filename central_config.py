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
from vertexai.preview.vision_models import Image, ImageGenerationModel
from vertexai.preview.generative_models import GenerativeModel, Part, GenerationConfig
import vertexai.preview.generative_models as generative_models
import pandas as pd
import time
import plotly.express as px
from vertexai.language_models import TextGenerationModel
from vertexai.preview.vision_models import Image, ImageGenerationModel
from vertexai.preview.generative_models import GenerativeModel, Part, grounding, Tool
import vertexai.preview.generative_models as generative_models
import os
import threading
import queue
import google.auth
from google.auth import transport
from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
import json
import io
from datetime import timedelta
import random
import datetime
import calendar
from faker import Faker  #need to install package
fake = Faker()
 
gs_folder = "your-gcp-bucket/"  # Replace with your actual bucket name, currently works with public gcp buckets
# will be updated to youtube video urls in the future.
bucket_name = "your-gcp-bucket"
region = "us-central1"


print("central config")
def keyload(): #keyloader allows local dev with a key.json file, but falls back to google auth for deployments.
    print("centralconfig keyload")
    try:
        print("try keyload from file")
        credentials, project_id = google.auth.load_credentials_from_file('../keys/key2.json')
        print("file loaded")
        return credentials, project_id
    except:
        print("command line auth?")
        credentials, project_id = google.auth.default()
        return credentials, project_id
credentials, project_id = keyload()

def load_config_file(c):
    config_file_path = c
    try:
        with open(config_file_path, "r") as f:
            config_data = json.load(f)
            return config_data  # Return the config_data directly
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"Error loading config: {e}")
        return None

config = load_config_file("data/config.json")

if config:
    print(config)
    demo_mode = config["demo_mode"]
    sleep_time = config["sleep_time"]
    logo_path = config["logo_path"]
    logo_width = config["logo_width"]
    company = config["company"]
    brand_names = config["brand_names"]
    region = config["region"]
    imgmodel = ImageGenerationModel.from_pretrained(config["imagen_version"])
    model = GenerativeModel(config["model"])
    customer_tags = config["customer_tags"]
    p41 = config["p41"]
    p42 = config["p42"]
    p43 = config["p43"]
    contentUrls = config["contentUrls"]
    focus_products = config["focus_products"]
else:
    st.write("error loading config file")



@st.cache_resource
def get_storage_client():
  return storage.Client()

@st.cache_resource
def get_bucket_from_storage_client():
  return get_storage_client().bucket(bucket_name)

def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""
    storage_client = get_storage_client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)

    # Note: The call returns a response only when the iterator is consumed.
    for blob in blobs:
        print(f"gs://{bucket_name}/{blob.name}")

def upload_file(bucket_name, source_file_name, destination_blob_name):
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    generation_match_precondition = 0
    blob.upload_from_filename(source_file_name, num_retries=2)
    out = f"File {source_file_name} uploaded to {destination_blob_name}."
    return(out)

def load_video(selected_video_url):
    """Loads and embeds the video."""
    video_placeholder = st.empty()
    video_placeholder.markdown(
    f"""
    <div style="display:flex; justify-content:center;">
        <video controls autoplay muted style="width:100%; max-height:500px">
            <source src="{selected_video_url}" type="video/mp4">
        </video>
    </div>
    """,
    unsafe_allow_html=True,
    )
    return video_placeholder

def vertex_text(textprompt):
    chat_text = "This text means there was a weird error."
    response = model.generate_content(
    textprompt,
    generation_config={
        "max_output_tokens": 8192,
        "temperature": 0.9,
        "top_p": 1
    },
    safety_settings={
          generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
          generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
          generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
          generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
    stream=False,
    )
    model_output = response.text #response data from vertex-text
    print('function vertext text generated response')
    return model_output

def display_additional_notes(json_results,propname):
    """Displays additional notes, handling different formats based on the input JSON."""
    additional_notes = json_results.get(f"{propname}", [])  # Get AdditionalNotes, defaulting to [] if not present

    if isinstance(additional_notes, list):  # List of dictionaries (mdlz_ad0)
        for note in additional_notes:
            if isinstance(note, dict):
                text = note.get('Text', "")
                st.markdown(text)
    elif isinstance(additional_notes, dict):  # Dictionary (mdlz_ad1, mdlz_ad2)
        # Check for "Text" key (mdlz_ad1) or directly use the dictionary (mdlz_ad2)
        text = additional_notes.get('Text', additional_notes)
        if isinstance(text, str):
            st.markdown(text)
        else:  # Handle the case where 'Text' contains a dictionary or other structure
            for key, value in text.items():
                st.markdown(f"**{key}:** {value}")
    else:
       st.markdown(additional_notes)
        
def genai_video_json_strict(uploaded_file, prompt, json_structure):
    print("starting analysis of video file")
    if uploaded_file is not None:
        print("generating results")
        generation_config=GenerationConfig(
            temperature=1.0,
            max_output_tokens=8192,
            top_p=.95,
            response_mime_type="application/json",
            response_schema=json_structure
        )
        safety_settings={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        stream=False
        print(f"-------------------prompt:")
        print(f"-------------------uploaded file: {uploaded_file}")
        response = model.generate_content([prompt,uploaded_file],generation_config=generation_config) #send prompt and file
        out = response.text
        return(out)
    
def vertex_text_strict(textprompt,json_structure):
    print("starting analysis of video file")
    print("generating results")
    generation_config=GenerationConfig(
        temperature=1.0,
        max_output_tokens=8192,
        top_p=.95,
        response_mime_type="application/json",
        response_schema=json_structure
    )
    safety_settings={
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
    stream=False
    print(f"-------------------prompt:")
    response = model.generate_content([textprompt],generation_config=generation_config) #send prompt and file
    out = response.text
    return(out)

def time_to_seconds(time_str):
    """Converts a time string in MM:SS format to total seconds."""
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

#######################################

styling = """ <style>
    body {
        background-color: white !important;
        color:rgb(38, 39, 48) !important;
        }
    header[data-testid="stHeader"]  {
        background-color:#fff !important;
        color:rgb(38, 39, 48) !important;
    }
    div[data-testid="stDeployButton"] {
        display: none !important;
    }
    .st-an{
        background-color:#f3f3f3 !important;
        color:rgb(38, 39, 48) !important;
    }
    .st-g0{
        color:black;
    }
    li[role="option"][aria-selected="false"] {
        background-color:#f3f3f3 !important;
        color:rgb(38, 39, 48) !important;
    }
    li[role="option"][aria-selected="true"] {
        background-color:#e3e3e3 !important;
        color:rgb(38, 39, 48) !important;
    }
    ul[role="listbox"] {
        background-color:gray;
    }
    textarea {
        background-color:#f3f3f3 !important;
        color:rgb(38, 39, 48) !important;
    }
    table {
        color:black !important;
    }
    .stApp, stSelectbox {
            background-color: white;
        }
    
    h1, h2, h3, h4, p {
        color:rgb(38, 39, 48) !important;
        }
    ul, li {
        color:black;
    }
    button, input{
        background-color:#f3f3f3 !important;
        color:black !important;
    }
    table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 10px;
        border: 2px solid #ddd;
        overflow: hidden;
    }
    th, td {
        padding: 8px;
        text-align: left;
    }
      .logo-container {
          display: flex; /* Use flexbox for alignment */
          justify-content: center; /* Center the logo horizontally */
          align-items: center; /* Center the logo vertically */
          margin-bottom: 20px; /* Add some space below the logo */
      }
      .logo {
          max-width: 200px; /* Adjust the maximum width as needed */
      }

    #row4_1 div[data-testid="column"] {
        background-color: white; /* Set column background to white (if needed) */
        padding: 10px;         /* Optional: add padding */
        border-radius: 5px;    /* Optional: rounded corners */
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2); 
    }
    </style>"""



#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################


#######################################################
#Configurations from curtis-genai-marketing

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

users_to_create = 10
url_list = [url.strip() for url in contentUrls.split(",")]
data ={}
anon = {}
alldata = []
lst = []
temp_dfs = []
chunk_size = 100

df = pd.DataFrame(index=range(users_to_create))

def store_global_dataframe(user_df):
    global lst
    lst.append(user_df)

def LeanFalse():
        spin = random.choices(
            population=['True','False'],
            weights=[.3,.7],
            k=1
            )
        output = (", ".join(spin))
        return(output)

def CustomerSpecificTopics(customer_tags):
    try:
        # Generate random weights that add up to 1
        num_tags = len(customer_tags)
        weights = [random.random() for _ in range(num_tags)]  # Generate random numbers
        weights_sum = sum(weights)
        weights = [w/weights_sum for w in weights]  # Normalize to sum to 1

        spin = random.choices(
            population=customer_tags,
            weights=weights,
            k=1
        )
        output = (", ".join(spin))
        return output
    except ValueError as err:
        print("Unexpected Error: ", err)
        raise

def LeanTrue():
        spin = random.choices(
            population=['True','False'],
            weights=[.7,.3],
            k=1
            )
        output = (", ".join(spin))
        return(output)

def runner():
    starting_uid = 1
    print("Starting User Creation at ",starting_uid)
    upper_bound = users_to_create + starting_uid
    user_range = list(range(starting_uid, upper_bound))
    for user in user_range:
        create_user(user)

def create_user(current_user):
    uid = current_user
    timestamp = ''
    timestamp = calendar.timegm((fake.date_time_between(start_date='-90d', end_date='now')).timetuple())
    full_name = fake.name()
    FLname = full_name.split(" ")
    Fname = FLname[0]
    Lname = FLname[1]
    name = Fname +" "+ Lname
    #email as a subset of name
    domain_name = "@demo.com"
    email_address = Lname + domain_name
    email_address = email_address.lower()
    phone_number = fake.phone_number()
    fake_city = fake.city()
    fake_country = fake.country()
    fake_address = fake.address()
    fake_zipcode = fake.zipcode()
    data["Email"] = email_address
    data["Engagement"] = random.randrange(1, 100)
    data["LTV"] = random.randrange(1, 1000)
    data[p41] = round(random.uniform(.01, .99),2)
    data[p42] = round(random.uniform(.01, .99),2)
    data[p43] = round(random.uniform(.01, .99),2)
    data["Top Channel"] = random.choice(['Social', 'Twitter','Google', 'Facebook', 'Email', 'Web', 'Mobile'])
    data["Recent Purchase"] = CustomerSpecificTopics(customer_tags)
    data["In Cart"] = CustomerSpecificTopics(customer_tags)
    data["Recent Url"] = random.choice(url_list)
    data["Location"] = fake.state()
    #data["country"] = fake_country
    #data["city"] = fake_city

    user_df = pd.DataFrame.from_records([data])
    store_global_dataframe(user_df)

runner() 
df = pd.concat(lst)
print(f"Global Data Frame:{df}")


results_queue = queue.Queue()

########################################################


def resize_and_overwrite(filepath, resize_percentage, quality=90):
    """Resizes an image by a percentage, optionally reduces quality, and overwrites the original file."""
    
    try:
        with Image.open(filepath) as img:
            new_width = int(img.width * resize_percentage / 100)
            new_height = int(img.height * resize_percentage / 100)
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save the resized image, overwriting the original
            if img_resized.format in ('JPEG', 'WebP'):
                img_resized.save(filepath, quality=quality)
            else:
                img_resized.save(filepath)

        print(f"Image resized and overwritten at {filepath}")

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except Exception as e:
        print(f"Error processing image: {e}")



#########################################################
def vertex_text(textprompt):
    chat_text = "This text means there was a weird error."
    response = model.generate_content(
    textprompt,
    generation_config={
        "max_output_tokens": 2048,
        "temperature": 0.4,
        "top_p": 1
    },
    safety_settings={
          generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
          generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
          generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
          generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
    stream=False,
    )
    model_output = response.text #response data from vertex-text
    print('function vertext text generated response')
    return model_output

#######################################

def upload_and_analyze_image(uploaded_file, prompt):

    if uploaded_file is not None:
        with st.container(border=False):
            row1_0, row1_1, row1_2, row1_3 = st.columns((1,6,6,1))
            with row1_1:
                st.write("Your Image")
                st.image(uploaded_file)
            # Get the image content (directly from the in-memory file object)
            image_content = uploaded_file.getvalue()
            image1 = base64.b64encode(image_content).decode('utf-8') #convert to vertex required base64
            image_ready = Part.from_data(image1, mime_type="image/jpeg") #use 'part' function from vertexai
            response = model.generate_content([prompt,image_ready]) #send prompt and image
            pet = response.text
            # Display the results
            with row1_2:
                st.write("Vertex AI Response:")
                st.write(pet)
            return(pet)


####################################

def vertex_text_grounded(textprompt):
    tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
    chat_text = "This text means there was a weird error."
    
    response = model.generate_content(textprompt,tools=[tool])
    
    model_output = response.text #response data from vertex-text
    print('function vertext text generated response')
    
    
    return model_output

###########################33
def vertex_text_threaded(textprompt,thread):
    chat_text = "This text means there was a weird error."
    response = model.generate_content(
    textprompt,
    generation_config={
        "max_output_tokens": 2048,
        "temperature": 0.4,
        "top_p": 1
    },
    safety_settings={
          generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
          generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
          generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
          generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
    stream=False,
    )
    model_output = response.text #response data from vertex-text
    
    try:
        file_path = os.path.join("data", f"marketing_brief{thread}.txt") # Define the file path to save the text (adjust as needed)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f: # Write the text data to the file
            f.write(response.text)
            print(f"Text data saved successfully to: {file_path}")
    except:
        st.write("Create a folder 'data' in the same spot as this file and give your user write to it")
        exit(1)
   
    print('function vertext text generated response')

    return


##########################################
def vertex_image(image_prompt,photo_features,imgname):
    
    z = 0
    #prompts for image gen.
    imageprompt = f"{image_prompt}{photo_features}"
    genreturns = imgmodel.generate_images(
            prompt = imageprompt,
            number_of_images=3,
            safety_filter_level="block_few",
            person_generation="allow_adults",
        )
    print(f"gen returns: {genreturns}")
    data_dir = "./data"
    for each in genreturns:
        print('entered genreturns for images')
        genreturns[z].save(f"./data/{z}{imgname}.jpg")
        print(f"Image{z} saved")
        image_path = f"./data/{z}{imgname}.jpg" 
        resize_percentage = 75
        quality = 90
        resize_and_overwrite(image_path, resize_percentage, quality)
        print('image resized')
        z+=1
    return genreturns
    

########################
############################

 
def display_images(img_list,imgfilename):
    z = 0
    data_dir = "./data"
    for each in img_list:
        img_list[z].save(f"./data/{z}-{imgfilename}.jpg")
        print(f"Image{z} saved")
        image_path = f"./data/{z}-{imgfilename}.jpg" 
        resize_percentage = 50
        quality = 50
        resize_and_overwrite(image_path, resize_percentage, quality)
        print('image resized')      
        z+=1
    return()
#####################################
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href_link = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download image below {file_label}</a>'
    return href_link

