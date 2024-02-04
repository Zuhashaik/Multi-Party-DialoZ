import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def capitalize_first_letter(word):
    if not word:
        return word  # Return an empty string or None as is
    return word[0].capitalize() + word[1:]
capitalize_first_letter('zuhair')

def get_conversation(us, speakers, emotions):
    conversation = ''''''
    for u, s, e in zip(us, speakers, emotions):
        conversation = conversation + f'''{capitalize_first_letter(s)} : {u} <{e}>\n'''
    return conversation

#The proposed data preprocessing meathod "split_concat".
def process_data(df):
    conversations = []
    main_speaker = []
    trigger_utterance =[]
    trigger_utterance_sp=[]
    main_sentence = []
    for us, speakers, emotions in zip(df['utterances'], df['speakers'], df['emotions']):
        conversation = f'''{get_conversation(us, speakers, emotions)}'''
        for u, s, e in zip(us, speakers, emotions):
            conversations.append(conversation)
            main_speaker.append(capitalize_first_letter(speakers[-1]))
            trigger_utterance.append(u)
            trigger_utterance_sp.append(s)
            main_sentence.append(us[-1])

    
    df = pd.DataFrame({
            'conversations' : conversations,
            'main_speaker' : main_speaker,
            'trigger_utterance' : trigger_utterance,
            'trigger_utterance_sp':trigger_utterance_sp,
            'main_sentence' : main_sentence
        })
    return df

def get_prompt(conversation, trigger_utterance, main_speaker, main_sentence ):
    prompt = f'''<|system|>In your role as an expert in sentiment and emotion analysis, your primary objective is to identify trigger utterances for emotion-flips in multi-party conversations (in hindi-english codemixed). Evaluate the provided dialogue by analyzing changes in emotions expressed by speakers through their utterances. Your task is to determine the accuracy of the hypothesis based on these emotional shifts.

<|Hypothesis|> The utterance <{trigger_utterance}> is a trigger for the emotion-flip in <{main_speaker}'s> response <{main_sentence}>

<|conversation|>
{conversation}
        
<|assistant|> The given Hypothesis is'''
    return prompt

def get_prompt_dataset_test(data):
    text = []
    for a,b,c,d in zip(data['conversations'], data['main_speaker'], data['trigger_utterance'],data['main_sentence']):
        prompt = get_prompt(a, b, c, d)
        text.append(prompt)

    test = pd.DataFrame({
        'text':text
    })
    return test

def pipelined_process(data_path):
    test_df = pd.read_json(data_path)
    test_df = process_data(test_df)
    processed_data = get_prompt_dataset_test(test_df)
    return processed_data
