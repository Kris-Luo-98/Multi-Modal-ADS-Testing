import time
from openai import OpenAI
import argparse
from datetime import datetime
import os
import base64
import pickle
import ast
import yaml
import pandas as pd


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_road_network(p1,p2,summary,sketch,model,record,folder_path):
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        # System prompt
                        "text": p1
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        # Task explanation
                        "text": p2
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please analyze the following case and provide your answer in the specified format.\n"
                    },
                    {
                        "type": "text",
                        "text": f"Summary:\n{summary}"
                    },
                    {
                        "type": "text",
                        "text": "\nSketch:\n"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{sketch}"
                        }
                    }
                ]
            }
        ],
        temperature=0,
        max_tokens=1280,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )

    first_output = response.choices[0].message.content
    # save results
    file_name = f"{record}_road_network.txt"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(first_output)

def extract_data_from_files(folder_path):
    data_dict = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            data_id = filename.split('_')[0]
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            try:
                potential_dicts = []
                start = 0
                while True:
                    start = content.find("{", start)
                    if start == -1:
                        break
                    end = content.find("}", start) + 1
                    if end == 0:
                        break
                    try:
                        possible_dict = ast.literal_eval(content[start:end])
                        if isinstance(possible_dict, dict):
                            potential_dicts.append(possible_dict)
                    except (ValueError, SyntaxError):
                        pass
                    start = end

                for d in potential_dicts:
                    if 'Validation' in d:
                        data_dict[data_id] = d
                        break
                else:
                    print(f"No valid structured data found in: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return data_dict


def save_data_to_pickle(data_dict, folder_path, filename="road_network_results.pkl"):
    pkl_filepath = os.path.join(folder_path, filename)
    with open(pkl_filepath, 'wb') as pkl_file:
        pickle.dump(data_dict, pkl_file)


def main():
    # project_path = os.path.dirname(os.path.dirname(os.getcwd()))
    project_path = r'/Users/kris/Desktop/Multi-Modal-ADS-Testing'
    parser = argparse.ArgumentParser(description='MM ADS Testing - road network extraction')
    parser.add_argument('--data_path', default=os.path.join(project_path, 'Crash_dataset'),
                        type=str, help='Path of MM crash dataset')
    parser.add_argument('--set_name', default='fullset', type=str,
                        help='Doing on which set?')
    # parser.add_argument('--road_type',
    #                     default=os.path.join(project_path, 'Information_extraction', 'Road_type',
    #                                          'results_2024-10-30_21-26-59', 'road_type_results.pkl'),
    #                     type=str, help='Path of road type results')
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

    # with open(args.road_type, 'rb') as f:
    #     pre_results = pickle.load(f)
    # road_type_labels = {ID: values[0] for ID, values in pre_results.items()}

    # Create results folder
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder = f"remove_rd_types_results_{current_time}"
    os.makedirs(result_folder, exist_ok=True)
    folder_path = os.path.abspath(result_folder)
    model = args.gpt

    for record in records:
        # Prepare record
        record_path = os.path.join(args.data_path, record)

        # Get encoded crash sketch
        sketch = encode_image(os.path.join(record_path, 'Sketch.jpg'))

        # Get crash summary
        with open(os.path.join(record_path, 'Summary.txt'), 'r', encoding='utf-8') as file:
            summary = file.read()

        # Prepare system prompt - Give roles and tasks
        # p1
        with open("./rd_prompts/p1.txt", 'r', encoding='utf-8') as file:
            p1 = file.read()

        # p2
        with open("./rd_prompts/p2.txt", 'r', encoding='utf-8') as file:
            p2 = file.read()

        extract_road_network(p1,p2,summary,sketch,model,record,folder_path)

        print(f'Record: {record} is finished!')
        time.sleep(2)

    # data_dict = extract_data_from_files(folder_path)
    # save_data_to_pickle(data_dict, folder_path)
    #
    # total_count = len(data_dict)
    # passed_count = sum(1 for record in data_dict.values() if record.get('Validation') == 'Passed')
    # pass_rate = (passed_count / total_count) * 100
    # print(f"Total case number: {len(records)}")
    # print(f"Extracted case number {total_count}")
    # print(f"Validation Pass Rate: {pass_rate:.2f}%")
    # output_info = {
    #     "Total case number": len(records),
    #     "Extracted case number": total_count,
    #     "Validation Pass Rate": f"{pass_rate:.2f}%"
    # }
    # pkl_filepath = folder_path + "/validation_info.pkl"
    # with open(pkl_filepath, 'wb') as pkl_file:
    #     pickle.dump(output_info, pkl_file)


if __name__ == '__main__':
    main()
