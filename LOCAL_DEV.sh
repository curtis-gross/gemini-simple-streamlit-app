#!/bin/bash
echo "Run this file with 'source' like this: source LOCAL_DEV.sh"
echo "Yes, you need a key file as well as a python virtual env created and populated"
export GOOGLE_APPLICATION_CREDENTIALS="../keys/key.json"
source ../../pipenv/bin/activate
echo "Now run : streamlit run image_analysis.py or whatever your main streamlit py file is"