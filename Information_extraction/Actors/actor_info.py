import pickle
from openai import OpenAI
import argparse
from datetime import datetime
import os
import pandas as pd
import base64
import re
import json
import pickle
import ast


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def read_trajectory_pkl(pkl_path):
    with open(pkl_path, "rb") as pkl_file:
        data = pickle.load(pkl_file)
        return str(data)


def get_traj(model,system,task,example_p1,example_p2,example_sketch,sketch,summary,road_attribute,record,folder_path):
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
                        "text": task
                    },
                    {
                        "type": "text",
                        "text": example_p1
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{example_sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": example_p2
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "I am equipped with skills in vehicle trajectory prediction and waypoint mapping on structured coordinate systems. Please provide a new case with a summary, crash sketch, and road network details, and I will extract accurate 2D vehicle trajectories accordingly."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's your output for this case?\nSketch:\n"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f'\nSummary: \n{summary}'
                    },
                    {
                        "type": "text",
                        "text": f'\nRoad Network: \n{str(road_attribute)}'
                    }
                ]
            }
        ],
        temperature=1,
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
    file_name = f"{record}_trajectory.txt"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(first_output)


def extract_trajectory_and_validation(file_content):
    traj_pattern = r'["\']V\d+_traj["\']: \[(.*?)\]'
    type_pattern = r'["\']V\d+_type["\']: ["\'](\w+)["\']'
    validation_pattern = r'["\']Validation["\']: ["\'](\w+)["\']'

    traj_matches = re.findall(traj_pattern, file_content)
    type_matches = re.findall(type_pattern, file_content)
    validation_match = re.search(validation_pattern, file_content)

    trajectory_data = {}
    for idx, traj in enumerate(traj_matches, start=1):
        traj_list = ast.literal_eval(f"[{traj}]")
        trajectory_data[f'V{idx}_traj'] = str(traj_list)
    for i in range(len(type_matches)):
        idx = i + 1
        trajectory_data[f'V{idx}_type'] = type_matches[i]

    validation_result = validation_match.group(1) if validation_match else None
    trajectory_data['Validation'] = validation_result

    return trajectory_data


def main():
    project_path = os.path.dirname(os.path.dirname(os.getcwd()))
    parser = argparse.ArgumentParser(description='MM ADS Testing - road network extraction')
    parser.add_argument('--data_path', default=os.path.join(project_path, 'Crash_dataset'), type=str,
                        help='Path of crash dataset.')
    parser.add_argument('--road_type',
                        default=os.path.join(project_path, 'Information_extraction', 'Road_type',
                                             'results_2024-10-30_21-26-59', 'road_type_results.pkl'),
                        type=str, help='Path of road type results')
    parser.add_argument('--road_network',
                        default=os.path.join(project_path, 'Information_extraction', 'Road_network',
                                             'results_2024-10-30_21-58-04', 'road_network_results.pkl'),
                        type=str,
                        help='Path of road type label file')
    parser.add_argument('--set_name', default='fullset', type=str,
                        help='Doing on validation or testing set?')
    parser.add_argument('--gpt', default=r'gpt-4o')
    args = parser.parse_args()
    Data_folder_path = os.path.join(project_path, 'Dataset_splitting')
    model = args.gpt

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

    with open(args.road_type, 'rb') as f:
        pre_results = pickle.load(f)
    road_type_labels = {ID: values[0] for ID, values in pre_results.items()}

    # Get road networks
    with open(args.road_network, 'rb') as file:
        road_network = pickle.load(file)

    # Create result folder
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder = f"results_{current_time}"
    os.makedirs(result_folder, exist_ok=True)
    folder_path = os.path.abspath(result_folder)

    # folder_path = r'C:\Users\Kris\Desktop\Multi-Modal-ADS-Testing\Information_extraction\Actors\results_2024-11-03_18-31-50'

    for record in records:
        # data preparation
        record_path = os.path.join(args.data_path, record)
        # Get encoded crash sketch
        sketch = encode_image(os.path.join(record_path, 'Sketch.jpg'))
        # Get crash summary
        with open(os.path.join(record_path, 'Summary.txt'), 'r', encoding='utf-8') as file:
            summary = file.read()
        road_type = road_type_labels[record]
        road_attribute = road_network[record]
        road_attribute['Road type'] = road_type
        example_sketch = encode_image(r'.\prompts\sketch_example_1.jpg')
        with open("./prompts/system.txt", 'r', encoding='utf-8') as file:
            system = file.read()
        with open("./prompts/task.txt", 'r', encoding='utf-8') as file:
            task = file.read()
        with open("./prompts/example_p1.txt", 'r', encoding='utf-8') as file:
            example_p1 = file.read()
        with open("./prompts/example_p2.txt", 'r', encoding='utf-8') as file:
            example_p2 = file.read()

        # Get trajectory from GPT
        get_traj(model,system,task,example_p1,example_p2,example_sketch,sketch,summary,road_attribute,record,folder_path)
        print(f'Record: {record} finished!')

    traj_result = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            record_id = file_name.split('_')[0]
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            output_index = max(content.find("Output"), content.find("output"))
            if output_index != -1:
                content = content[output_index + len("Output"):].strip()

            x = extract_trajectory_and_validation(content)
            traj_result[str(record_id)] = x

    data_path = os.path.join(folder_path,'actor_results.pkl')
    with open(data_path, 'wb') as pkl_file:
        pickle.dump(traj_result, pkl_file)

    total_count = len(traj_result)
    passed_count = sum(1 for record in traj_result.values() if record.get('Validation') == 'Passed')
    pass_rate = (passed_count / total_count) * 100
    print(f"Total case number: {len(records)}")
    print(f"Extracted case number {total_count}")
    print(f"Validation Pass Rate: {pass_rate:.2f}%")
    output_info = {
        "Total case number": len(records),
        "Extracted case number": total_count,
        "Validation Pass Rate": f"{pass_rate:.2f}%"
    }
    pkl_filepath = os.path.join(folder_path,'validation_info.pkl')
    with open(pkl_filepath, 'wb') as pkl_file:
        pickle.dump(output_info, pkl_file)


if __name__ == '__main__':
    main()
