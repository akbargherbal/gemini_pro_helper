import base64
import vertexai
from vertexai.generative_models import GenerativeModel, FinishReason
# import vertexai.preview.generative_models as generative_models
from vertexai import generative_models

import os
import logging
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from time import sleep
import threading



def configure_logging():
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f'labeling_biocon_sentences_{time_stamp}.log'
    logging.basicConfig(filename=log_file_name, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

configure_logging()

# Create a lock to coordinate the sleep
sleep_lock = threading.Lock()

# CONSTANTS
TIME_DELAY = 0.3
time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

df_result = pd.DataFrame(data={'result': [], 'packet':[]})
df_result_file_name = f'./df_result_{time_stamp}.pkl'

df_result.to_pickle(df_result_file_name, protocol=4)

google_sev_acc_path = './gemini_service_account.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_sev_acc_path

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

generation_config = {
    "max_output_tokens": 8150,
    "temperature": 0.5,
    "top_p": 0.95,
}

region_zone = 'us-central1'
vertexai.init(project="gen-lang-client-0867780601", location=region_zone)
model = GenerativeModel("gemini-1.0-pro")




prompt = '''
Your goal is to assist simultaneous interpreters in preparing for an upcoming event by providing them with realistic and engaging transcripts of various sessions. Imagine You have access to the entire event's transcript and details about each session, including the agenda, session titles, and brief descriptions. Based on this information, you'll generate transcripts that simulate the flow of conversation and interaction between participants, ensuring they reflect the natural interactions among speakers. Aim for a conversational tone, using simple language suitable for interpreters with varying levels of English proficiency. Avoid unnecessary complexity and overly formal language, focusing on clarity and accessibility. Aim for a transcript length of 5000-8100 words (40-55 minutes).

Instructions:
1.	Read the provided context: This will include the overall agenda of the event and details about specific sessions.
2.	Analyze the question: The question will specify the particular session for which you need to generate a transcript.
3.	Generate the transcript:
-	Create names: Invent plausible names for the speakers, moderator (if applicable), and any mentioned organizations or entities. Aim for names that reflect the professional context and potential nationalities of the participants.
-	Imagine the speakers as experts relevant to the session topic, each with unique backgrounds and perspectives.
-	Based on the session title and description, determine the most likely format (e.g., panel discussion, presentation, Q&A).
-	Utilize the session description to identify key discussion points and potential talking points.
-	Expand on these points by incorporating relevant examples, case studies, or current events.
-	Explore different viewpoints and potential areas of debate or agreement among the speakers.
-	Create natural and engaging dialogue that reflects the speakers' personalities and expertise.
-	Include moments of interaction, such as questions, interruptions, and agreements/disagreements.
-	Maintain a conversational tone, using clear and simple language accessible to interpreters.
-	Structure the transcript to reflect the chosen session format, including introductions, presentations (if applicable), moderated discussions, audience interaction, and concluding remarks.
'''.strip()


def get_context(file):
    with open(file, encoding='utf-8') as f:
        context = f.read()
    return context

def create_prompt(data, context, prompt=prompt):
    result = f"""
{prompt}

###### CONTEXT START ######
{context}
###### CONTEXT END ######

###### QUESTION START ######
{data[0]}
{data[1]}
{data[2]}
###### QUESTION END ######
""".strip()
    return result

def generate(data, context, packet_index):
    try:
        question = create_prompt(data, context)
        responses = model.generate_content(question, generation_config=generation_config,
            safety_settings=safety_settings)
        return responses,packet_index
    except Exception as e:
        print(e)
        logging.error(f'Error: {packet_index} {e}')
        return None,packet_index
    finally:
        with sleep_lock:
            print(f"Sleeping for {TIME_DELAY} seconds")
            logging.info(f"Sleeping for {TIME_DELAY} seconds")
            sleep(TIME_DELAY)

def process_file(executor, list_packets, context):
    # list_packets = list_packets[:5]
    failed_packets = []
    list_result = []
    for packet in list_packets:
        packet_index = packet[0]
        data = packet[1]

        future = executor.submit(generate, data, context, packet_index)
        result, packet = future.result()
        if result is None:
            failed_packets.append(packet)
        else:
            list_result.append({'result': result, 'packet': packet})
        

        # Create DataFrame from the list of results
        df_temp = pd.DataFrame(list_result)

        # Read existing DataFrame from pickle file
        df_main = pd.read_pickle(df_result_file_name)

        # Concatenate the new DataFrame with the existing one
        df_main = pd.concat([df_main, df_temp], ignore_index=True)
        df_main = df_main.drop_duplicates(subset=['packet']).reset_index(drop=True)


        # Save the updated DataFrame to pickle file
        df_main.to_pickle(df_result_file_name, protocol=4)

        print(f'Total files: {len(df_main)}')
        print(f'Total failed files: {len(failed_packets)}')

    logging.info(f'Total files: {len(df_main)}')
    df_main = df_main.drop_duplicates(subset=['packet']).reset_index(drop=True)
    return df_main, failed_packets

def main():
    list_packet = pd.read_pickle('./data_list.pkl')
    list_packet = list(zip(list_packet.IDX_PACKET,list_packet.DATA_LIST))
    context = get_context('./agenda.md')

    executor = ThreadPoolExecutor(max_workers=120)
    df_final, failed_files = process_file(executor, list_packet, context=context)

    df_final.to_pickle(f'./gemini_final_result_{time_stamp}.pkl', protocol=4)
    with open(f'./failed_files_{time_stamp}.txt', mode='w', encoding='utf-8') as f:
        for file in failed_files:
            f.write(file + '\n')
    

if __name__ == "__main__":
    main()
