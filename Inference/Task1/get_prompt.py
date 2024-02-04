import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def capitalize_first_letter(word):
    if not word:
        return word  # Return an empty string or None as is
    return word[0].capitalize() + word[1:]
capitalize_first_letter('zuhair')

#The proposed data preprocessing meathod "split_concat".
def split_concat(h, df):
    txt = []
    main_sentence = []
    main_sentence_speaker = []
    for utterances, speakers in zip(df['utterances'], df['speakers']):
        talk = []
        for utterance, speaker in zip(utterances, speakers):
            speaker = capitalize_first_letter(speaker)
            if speaker is not None:
                utterance = speaker + ' : ' + utterance
            main_sentence.append(utterance)
            main_sentence_speaker.append(speaker)
            if len(talk) > h-1 : # It make sure that conversation has the height of sentext height or less than that if data is
                talk.pop(0)
            txt.append('\n'.join(talk) + '\n' + utterance)
            talk.append(utterance)

    # This removes '\n' from the starting of the sentence that doesnt affect the prompts for llms
    txt = [conversation[1:] if conversation.startswith('\n') else conversation for conversation in txt]

    new_df = pd.DataFrame(
        {
            'sentext_conversation': txt,
            'present_sentence':main_sentence,
            'present_sentence_speaker':main_sentence_speaker
            #last sentence is the main sentence that speaker's emotion to be finded!
        }
    )
    return new_df

def get_prompt(conversation, speaker, main_sentence):
    prompt = f'''<|system|>You are an expert in sentiment and emotional analysis, find the emotion of the utterance in the given conversation (in hindi-english code mixed) from these classes, [anger, contempt, disgust, fear, joy, neutral, sadness, surprise].
<|utterance|> {speaker} : {main_sentence}
<|conversation|>
{conversation}
<|assistant|>The emotion is :'''
    return prompt

def get_prompt_dataset_test(data):
    text = []
    for n,conversation in enumerate(data['sentext_conversation']):
        prompt = get_prompt(conversation, data['present_sentence_speaker'][n], data['present_sentence'][n])
        text.append(prompt)
    dataset = pd.DataFrame({
        'text':text
        })
    return dataset

def pipelined_process(data_path, sentext_height=3):
    test_df = pd.read_json(data_path)
    test_df = split_concat(sentext_height, test_df)
    processed_data = get_prompt_dataset_test(test_df)
    return processed_data
