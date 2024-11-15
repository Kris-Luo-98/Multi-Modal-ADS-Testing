from openai import OpenAI
import argparse
from datetime import datetime
import os
import re
import ast
import pickle
import json


def extract_info(content):
    extracted_info = {"Weather": None, "Time": None, "Validation": None}
    content_str = content if isinstance(content, str) else json.dumps(content)
    weather_match = re.search(r'"Weather":\s*"([^"]*)"|Weather:\s*["“](Sunny|Cloudy|Clear|Rainy|Foggy|Snowy|Windy)["”]', content_str, re.IGNORECASE)
    time_match = re.search(r'"Time":\s*"([^"]*)"|Time:\s*["“](Daytime|Nighttime)["”]', content_str, re.IGNORECASE)
    validation_match = re.search(r'"Validation":\s*"([^"]*)"|Validation:\s*["“](Passed|Failed)["”]', content_str, re.IGNORECASE)
    if weather_match:
        extracted_info["Weather"] = weather_match.group(1) or weather_match.group(2)
    if time_match:
        extracted_info["Time"] = time_match.group(1) or time_match.group(2)
    if validation_match:
        extracted_info["Validation"] = validation_match.group(1) or validation_match.group(2)

    return extracted_info


def get_env(system,p1,p2, record, summary,folder_path,model):
    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": p1
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Sure, please go ahead and provide the example so I can better understand the task and assist you accordingly."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": p2
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Thank you for the example, I have understood the task, please give me the data."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"What's your output for this summary:\n{summary}"
                    }
                ]
            }
        ],
        temperature=0,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )

    first_output = response.choices[0].message.content
    file_name = f"{record}_env_info.txt"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(first_output)

def main():
    project_path = os.path.dirname(os.path.dirname(os.getcwd()))
    parser = argparse.ArgumentParser(description='MM ADS Testing - road type extraction')
    parser.add_argument('--data_path', default=os.path.join(project_path, 'Crash_dataset'), type=str,
                        help='Path of crash dataset.')
    parser.add_argument('--set_name', default='fullset', type=str,
                        help='Doing on which set?')
    parser.add_argument('--gpt', default=r'chatgpt-4o-latest')
    args = parser.parse_args()
    Data_folder_path = os.path.join(project_path, 'Dataset_splitting')

    # Get data ID list
    if args.set_name == 'fullset':
        set_path = os.path.join(Data_folder_path, 'full_set.pkl')
        print('Work on full dataset!')
    else:
        set_path = os.path.join(Data_folder_path,
                                'validation_set.pkl') if args.set_name == 'validation' else os.path.join(
            Data_folder_path,
            'testing_set.pkl')
        print(f'Work on {args.set_name}!')

    with open(set_path, 'rb') as file:
        records = pickle.load(file)

    # Create result folder
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder = f"results_{current_time}"
    os.makedirs(result_folder, exist_ok=True)
    folder_path = os.path.abspath(result_folder)

    for record in records:
        record_path = os.path.join(args.data_path, record)
        # Get crash summary
        with open(os.path.join(record_path, 'Summary.txt'), 'r', encoding='utf-8') as file:
            summary = file.read()

        # Prepare prompts
        with open('./prompts/system.txt', 'r', encoding='utf-8') as file:
            system = file.read()
        with open('./prompts/p1.txt', 'r', encoding='utf-8') as file:
            p1 = file.read()
        with open('./prompts/p2.txt', 'r', encoding='utf-8') as file:
            p2 = file.read()

        # Get response from GPT
        get_env(system,p1,p2,record, summary,folder_path,args.gpt)
        print(f'Record: {record} finished!')

    data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('_env_info.txt'):
            file_id = filename.split('_')[0]

            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                data[file_id] = extract_info(content)

    data_path = folder_path + '/env_info.pkl'
    with open(data_path, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)

    total_count = len(data)
    passed_count = sum(1 for record in data.values() if record.get('Validation') == 'Passed')
    pass_rate = (passed_count / total_count) * 100
    print(f"Total case number: {len(records)}")
    print(f"Extracted case number {total_count}")
    print(f"Validation Pass Rate: {pass_rate:.2f}%")
    output_info = {
        "Total case number": len(records),
        "Extracted case number": total_count,
        "Validation Pass Rate": f"{pass_rate:.2f}%"
    }
    pkl_filepath = folder_path + "/validation_info.pkl"
    with open(pkl_filepath, 'wb') as pkl_file:
        pickle.dump(output_info, pkl_file)


if __name__=='__main__':
    main()