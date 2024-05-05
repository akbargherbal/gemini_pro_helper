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
TIME_DELAY = 12.5
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
model = GenerativeModel("gemini-1.5-pro-preview-0409")




prompt = '''
Read the attached PDF document, then list both the positive and negative aspects of the country portfolio.
'''.strip()



def create_prompt(data, prompt=prompt):
    result = f"""
{prompt}

###### QUESTION START ######
{data}
###### QUESTION END ######
""".strip()
    return result

def generate(data, packet_index):
    try:
        question = create_prompt(data)
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

def process_file(executor, list_packets):
    # list_packets = list_packets[:2]
    failed_packets = []
    list_result = []
    for packet in list_packets:
        packet_index = packet[0]
        data = packet[1]

        future = executor.submit(generate, data, packet_index)
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
    list_packet = pd.read_pickle('./COUNTRIES_LIST.pkl')
    list_packet = list(zip(list_packet.IDX_PACKET,list_packet.TEXT))
    

    executor = ThreadPoolExecutor(max_workers=120)
    df_final, failed_files = process_file(executor, list_packet)

    df_final.to_pickle(f'./gemini_final_result_{time_stamp}.pkl', protocol=4)
    with open(f'./failed_files_{time_stamp}.txt', mode='w', encoding='utf-8') as f:
        for file in failed_files:
            f.write(file + '\n')
    

if __name__ == "__main__":
    main()
