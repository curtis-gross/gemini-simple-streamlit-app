# Starter App
This Image Analysis application is meant to be as simple as possible to get a cloud run + gemini application runing.
Note that the pre-work is getting a google cloud project up and running, in the future I am going to look to create scripts to make that process easier.
Additonally note that I included a few video analysis files as well, these take more work because not only do you need vertex, but a gcp bucket (if you have mp4/video files) or a change to this code to support youtube links directly (wip on my side as well for the future)

# Instructions
- You will need to get your local development environment configured to work with gcloud
- https://cloud.google.com/sdk/docs/install-sdk
- Install Python
- Create a Python virtual environment
- Install Streamlit and the other requirements (in requirements.txt)
- Modify Procfile
- change the cloudrun.sh file to be runnable. chmod 755
- Run in your terminal, streamlit run image_analysis.py

The app:
- There is a default prompt, output prompt and default image. 
- You can press 'go' to see the results or upload your own image.

Messy and more advanced:
- the video ad analysis and Audiece / image creation pages are less clean, they are fun tech demos but I have had less time to convert the code to something sharable, so it is a bit more of a mess.  I will spend time cleaning up those examples in the future.
