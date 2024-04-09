import streamlit as st
import pandas as pd
import requests
import logging
import datetime

# Initialization
st.set_page_config(page_title="Fish Detection App", layout="wide", page_icon=":fish:")
logging.basicConfig(level=logging.INFO, filename='front_end_logs.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Styling
st.markdown("""
<style>
footer {visibility: hidden;}
section.main footer {visibility: visible;}
div.streamlit-expander {padding: 10px; background-color: #f1f1f1; border: 1px solid #e1e1e1;}
button { /* Your button styles */ }
</style>
""", unsafe_allow_html=True)

BACKEND_URL = "https://sleepy-sands-32273-3dcb603050d0.herokuapp.com/"

def send_files_to_backend(uploaded_files):
    logging.info(f"Sending {len(uploaded_files)} files to backend for processing.")
    files = [("files", (file.name, file, "video/mp4")) for file in uploaded_files]
    try:
        response = requests.post(BACKEND_URL, files=files)
        if response.status_code == 200:
            logging.info("Files processed successfully.")
            return response.json()
        else:
            logging.error(f"Error processing files with status code {response.status_code}.")
            st.error(f"Failed to process files. Server responded with status code {response.status_code}. Please try again.")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        st.error("Failed to connect to the backend. Please check if the backend is running and try again.")
        return None
    except ValueError:
        logging.error("Failed to decode JSON from response.")
        st.error("Received an invalid response from the backend. Please try again.")
        return None


def log_wrong_classification(video_name):
    """Logs the video name of wrongly classified videos."""
    with open("wrong_classification.txt", "a") as log_file:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{now} - {video_name}\n")

def display_results(video_details):
    if 'video_details' not in st.session_state or not st.session_state.video_details:
        st.session_state.video_details = video_details['video_details']

    df_videos = pd.DataFrame(st.session_state.video_details).drop(columns=['Image Path'], errors='ignore')
    st.write(df_videos)

    for index, detail in enumerate(st.session_state.video_details):
        with st.expander(f"{detail['Name']}"):
            st.write(f"Label: {detail.get('Label', 'No Data')}")
            st.write(f"Fish Count: {detail.get('Fish Count', 'N/A')}")
            st.write(f"Direction: {detail.get('Direction', 'N/A')}")

            if detail.get('Image Path'):
                st.image(detail['Image Path'], caption="Detected Fish")

            # Generate a unique key for each checkbox by combining the video name and index
            unique_key = f"wrong_class_{detail['Name']}_{index}"
            wrong_classification = st.checkbox("Mark as incorrectly classified", key=unique_key)
            if wrong_classification:
                log_wrong_classification(detail['Name'])
                st.success(f"Marked {detail['Name']} as incorrectly classified.")


def main():
    st.title("🐟 Fish Detection Web App")
    uploaded_files = st.sidebar.file_uploader("Upload video files", type=["mp4", "avi"], accept_multiple_files=True, key="file_uploader")
    
    # Button to clear all outputs
    if st.sidebar.button("Clear All Outputs"):
        # Reset or remove the relevant session state variable
        if 'video_details' in st.session_state:
            del st.session_state.video_details
        st.experimental_rerun()

    if uploaded_files and st.sidebar.button("Start Processing"):
        video_details = send_files_to_backend(uploaded_files)
        if video_details:
            st.session_state.video_details = video_details['video_details']
            display_results(video_details)
        else:
            st.error("An error occurred. No details to display. Please check if files were correctly processed or try again.")
    elif "video_details" in st.session_state and st.session_state.video_details:
        display_results({"video_details": st.session_state.video_details})

    with st.expander("Privacy Concerns"):
        st.markdown("""
            ## Privacy Policy and Data Logging Information

            We are committed to protecting your privacy and the security of your data. Our system logs certain information to improve the functionality and performance of the application:

            - **Uploaded Videos**: Videos you upload are processed to detect fish presence. They are temporarily stored during the processing time and are not shared with any third parties.

            - **Processing Time**: We log the time it takes to process videos to ensure timely performance and identify potential areas for optimization.

            - **Usage Metrics**: We monitor user interactions with the application to improve usability and user experience.

            Please note that all logged data is used solely for the purpose of enhancing the application and is handled in accordance with data protection regulations.

            By using this application, you consent to the data practices described in this policy.""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
