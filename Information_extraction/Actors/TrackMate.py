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
import requests
import time
import random

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def read_trajectory_pkl(pkl_path):
    with open(pkl_path, "rb") as pkl_file:
        data = pickle.load(pkl_file)
        return str(data)


def get_traj(sketch,summary,road_network,gpt,record,folder_path):
    YOUR_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJwd2RfYXV0aF90aW1lIjoxNzI5NzQyMTc3ODM1LCJzZXNzaW9uX2lkIjoickhIb2Rvc09qUHlIYTZOYjFtTWVZa1pCb3hEQ2hNQnIiLCJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJrcmlzX2x1bzIwMjJAb3V0bG9vay5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZX0sImh0dHBzOi8vYXBpLm9wZW5haS5jb20vYXV0aCI6eyJwb2lkIjoib3JnLUhkMEtBTXVxcHF2RFNGeldhV3N5b2ZlcyIsInVzZXJfaWQiOiJ1c2VyLUpSSm92N3JUanloWWR3aEJWMzRIS1JoeCJ9LCJpc3MiOiJodHRwczovL2F1dGgwLm9wZW5haS5jb20vIiwic3ViIjoiYXV0aDB8NjNmMzZhODU2MmY1ZGE2ZWYyN2RiZDk4IiwiYXVkIjpbImh0dHBzOi8vYXBpLm9wZW5haS5jb20vdjEiLCJodHRwczovL29wZW5haS5vcGVuYWkuYXV0aDBhcHAuY29tL3VzZXJpbmZvIl0sImlhdCI6MTcyOTc0MjE3OSwiZXhwIjoxNzMwNjA2MTc5LCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIG1vZGVsLnJlYWQgbW9kZWwucmVxdWVzdCBvcmdhbml6YXRpb24ucmVhZCBvcmdhbml6YXRpb24ud3JpdGUgb2ZmbGluZV9hY2Nlc3MiLCJhenAiOiJUZEpJY2JlMTZXb1RIdE45NW55eXdoNUU0eU9vNkl0RyJ9.COlEvd9KXBNJKRXosMPdU_K_NGjAU4tXzBK2ENnp3bVF7NzI2DFKBRvv-tEGPryl-xRZcVgTxoEf52Q7FdEbgI7pR-jdQuo1hRkLQtEeygDsmNuv__WPKFstkqjERGrbHHZ8RAbO2-UCBoUWDSdi0qpeTyd3bFxXhjuSXCOxasurHC_9x-4LSVXwWMdKaqWJFw4SAw8wBt9GmoE9qogmRsKiPUcvGr9EEqBDT27dIP8CGjpmaT2b6KCU4iJBuO7cPJVqVx25g1a6V_675xrKajgPKF3flC2cDlBwy4DpwyiJjC86suykpxU7pQ1oHoCWrr6Upc2igfz6gJaGRZ_GbQ"

    url = "http://127.0.0.1:5005/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {YOUR_TOKEN}"
    }
    data = {
        "model": gpt,
        "messages": [
            {
            "role": "user",
            "content": [
                    {
                        "type": "text",
                        "text": "Hi Track Mate, what's your output for this case?\nSketch:\n"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"\nSummary:\n{summary}"
                    },
                    {
                        "type": "text",
                        "text": f"\nRoad Network:\n{str(road_network)}"
                    }
                ]
        }
        ],
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(data), stream=False)

    response_json = response.json()
    message_content = response_json['choices'][0]['message']['content']
    file_name = f"{record}_env_info.txt"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(message_content)
    random_number = random.randint(2, 7)
    time.sleep(random_number)

def extract_trajectory_and_validation(file_content):
    traj_pattern = r'["\']V\d+_traj["\']: \[(.*?)\]'
    type_pattern = r'["\']V\d+_type["\']: ["\'](\w+)["\']'
    validation_pattern = r'["\']Validation["\']: ["\'](\w+)["\']'

    traj_matches = re.findall(traj_pattern, file_content)
    type_matches = re.findall(type_pattern, file_content)
    validation_match = re.search(validation_pattern, file_content)

    trajectory_data = {}
    for idx, traj in enumerate(traj_matches, start=1):
        # 使用ast.literal_eval来安全地解析字符串
        traj_list = ast.literal_eval(f"[{traj}]")
        # trajectory_data[f'V{idx}_traj'] = traj_list
        trajectory_data[f'V{idx}_traj'] = str(traj_list)
    for i in range(len(type_matches)):
        idx = i + 1
        trajectory_data[f'V{idx}_type'] = type_matches[i]

    validation_result = validation_match.group(1) if validation_match else None
    trajectory_data['Validation'] = validation_result

    return trajectory_data
def main():
    base_dir = r'C:\Users\Kris\Desktop\MM-ADS-Testing\NHTSA'
    parser = argparse.ArgumentParser(description='MM ADS Testing - road network extraction')
    parser.add_argument('--data_path', default=r'C:\Users\Kris\Desktop\MM-ADS-Testing\NHTSA\Dataset', type=str,
                        help='Path of MM crash dataset')
    parser.add_argument('--Gpt_model', default=r'gpt-4-gizmo-g-84pJuqHxd', type=str)
    parser.add_argument('--road_type_label', default=r'C:\Users\Kris\Desktop\MM-ADS-Testing\Information Extraction\Road Type Extraction\road_type_extract_experiment_2024-10-29_14-28-55\road_type_prediction.xlsx', type=str,
                        help='Path of road type label file')
    parser.add_argument('--road_network',
                        default=r'C:\Users\Kris\Desktop\MM-ADS-Testing\Information Extraction\Road Network Extraction\road_network_extract_experiment_2024-10-29_14-56-21\road_network_results.pkl',
                        type=str,
                        help='Path of road type label file')
    parser.add_argument('--set_name', default='training', type=str,
                        help='Doing on training or testing set?')
    # parser.add_argument('--trajectory_training_data',
    #                     default=r".\Nuplan\trajectory",
    #                     type=str,
    #                     help='Path of pracjectory training data folder')
    # parser.add_argument('--trajectory_vis',
    #                     default=r".\Nuplan\trajectory visualization",
    #                     type=str,
    #                     help='Path of pracjectory visualization data folder')

    args = parser.parse_args()
    model = args.Gpt_model

    # Get data ID list
    set_path = os.path.join(base_dir, 'training_set.pkl') if args.set_name == 'training' else os.path.join(base_dir,
                                                                                                           'testing_set.pkl')
    with open(set_path, 'rb') as file:
        records = pickle.load(file)

    # Get road type labels
    df = pd.read_excel(args.road_type_label)
    road_type_labels = dict(zip(df['ID'], df['Label']))

    # Get road networks
    with open(args.road_network, 'rb') as file:
        road_network = pickle.load(file)

    # Get training data
    # trajectory_vis_path = args.trajectory_vis
    # trajectory_training_data_path = args.trajectory_training_data

    # Initialize lists
    # encoded_images = []
    # trajectory_data = []
    #
    # # Process images
    # image_files = sorted([f for f in os.listdir(trajectory_vis_path) if f.endswith('.jpg')])
    # for image_file in image_files:
    #     image_id = int(image_file.split('_')[-1].split('.')[0])
    #     image_full_path = os.path.join(trajectory_vis_path, image_file)
    #     encoded_images.append((image_id, encode_image(image_full_path)))
    #
    # # Sort encoded images by ID
    # encoded_images.sort(key=lambda x: x[0])
    # encoded_images = [img[1] for img in encoded_images]
    #
    # # Process PKL files
    # pkl_files = sorted([f for f in os.listdir(trajectory_training_data_path) if f.endswith('.pkl')])
    # for pkl_file in pkl_files:
    #     try:
    #         pkl_id = int(pkl_file.split('_')[-1].split('.')[0])
    #         pkl_full_path = os.path.join(trajectory_training_data_path, pkl_file)
    #         trajectory_data.append((pkl_id, read_trajectory_pkl(pkl_full_path)))
    #     except Exception as e:
    #         print(f"Error reading {pkl_file}: {e}")
    #
    # # Sort trajectory data by ID
    # trajectory_data.sort(key=lambda x: x[0])
    # trajectory_data = [data[1] for data in trajectory_data]

    # Create result folder
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder = f"trajectory_extract_experiment_{current_time}"
    os.makedirs(result_folder, exist_ok=True)
    folder_path = os.path.abspath(result_folder)

    for record in records:
        # data preparation
        record_id = int(record)
        record_path = args.data_path + '\\' + record

        # Get encoded crash sketch
        sketch = encode_image(record_path + '\\' + 'Sketch.jpg')

        # Get crash summary
        with open(record_path + '\\' + 'Summary.txt', 'r', encoding='utf-8') as file:
            summary = file.read()

        road_type = road_type_labels[record_id]

        road_attribute = road_network[record]
        road_attribute['Road type'] = road_type

        # example_sketch = encode_image(r'.\Few_shot_learning\sketch_example_1.jpg')
        # Get trajectory from GPT
        # if road_type == 'Curve':
        #     curve_1 = encode_image(r'E:\GitHub\MM-ADS-Testing\Scenario Reconstruction\Curve_road_margins_1.jpg')
        #     curve_2 = encode_image(r'E:\GitHub\MM-ADS-Testing\Scenario Reconstruction\Curve_road_margins_2.jpg')
        #     example_sketch = encode_image(r'.\Few_shot_learning\sketch_example_2.jpg')
        #     get_curve_traj(trajectory_data, encoded_images, example_sketch, sketch, summary, road_attribute, record, folder_path,
        #          model,curve_1,curve_2)
        # else:
        get_traj(sketch,summary,road_network,model,record,folder_path)
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

    data_path = folder_path + '\\traj_results.pkl'
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
    pkl_filepath = folder_path + "\\validation_info.pkl"
    with open(pkl_filepath, 'wb') as pkl_file:
        pickle.dump(output_info, pkl_file)





if __name__ == '__main__':
    main()
