# Transcript Generation Script

This Python script is designed to generate realistic and engaging transcripts for simultaneous interpreters, simulating the flow of conversation and interaction between participants in various sessions of an upcoming event.

## Prerequisites

The following files should be available in the same directory as the script:

- `gemini_service_account.json`: Google service account credentials file.
- `agenda.md`: A Markdown file containing the overall agenda of the event and details about specific sessions.
- `data_list.pkl`: A pickle file containing a list of session details to generate transcripts for.

## Usage

1. Install the required Python packages:
  - `vertexai`
  - `pandas`

2. Update the `google_sev_acc_path` variable with the path to your Google service account credentials file.

3. Run the script

The script will generate transcripts for the specified sessions, based on the provided context and session details. The generated transcripts will be saved in a pickle file named `gemini_final_result_{timestamp}.pkl`, where `{timestamp}` is the current date and time.

## Output

The script generates the following output files:

- `gemini_final_result_{timestamp}.pkl`: A pickle file containing the generated transcripts for each session.
- `failed_files_{timestamp}.txt`: A text file listing any sessions for which the transcript generation failed.
- `df_result_{timestamp}.pkl`: A pickle file containing intermediate results during the script execution.
- `labeling_biocon_sentences_{timestamp}.log`: A log file containing information about the script's execution.

## Script Overview

The script performs the following tasks:

1. Configures logging for the script's execution.
2. Reads the context (agenda and session details) from the `agenda.md` file.
3. Loads the list of sessions from the `data_list.pkl` file.
4. Initializes the VertexAI GenerativeModel for transcript generation.
5. Processes each session in parallel using a ThreadPoolExecutor.
6. For each session, generates a prompt based on the provided context and session details.
7. Generates the transcript using the VertexAI GenerativeModel.
8. Saves the generated transcripts and handles any failures during the generation process.
9. Writes the final results to the output files.

Note: The script uses the VertexAI GenerativeModel to generate the transcripts, which may incur costs based on your usage and pricing plan.