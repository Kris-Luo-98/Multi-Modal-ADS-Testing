import time
from openai import OpenAI
import argparse
from datetime import datetime
import os
import pandas as pd
import base64
import re
import pickle


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def road_type_extraction(sketch, summary, road_type_sketch, road_type_summary, p1, p2, client, record, folder_path,
                         model):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an experienced road engineering expert who is skilled in identifying road types[Straight, Curve, Intersection, T-intersection, Merge] from map sketches and vehicle behavior description."
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
                        "text": "Example data:\nSketch:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{road_type_sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"\nSummary: \n{road_type_summary} \n"
                    },
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
                        "text": "Got it! Please provide the crash details (sketch and summary) for the next case, "
                                "and I'll follow the process to extract the road type and validate it."
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
                        "text": f"\nSummary: \n{summary}"
                    }
                ]
            }
        ],
        # control creativity and randomness
        temperature=0,
        max_tokens=512,  # control output length
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )

    first_output = response.choices[0].message.content

    # save results
    file_name = f"{record}_road_type.txt"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(first_output)


def read_txt_results(folder_path):
    results = {}
    for filename in os.listdir(folder_path):
        if filename.endswith("_road_type.txt"):
            file_id = filename.split("_")[0]
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            road_type = re.search(r"'Road type': ([\w-]+)", content).group(1)
            validation = re.search(r"'Validation': (\w+)", content).group(1)
            results[file_id] = [road_type, validation]
    return results


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
        set_path = os.path.join(Data_folder_path, 'validation_set.pkl') if args.set_name == 'validation' else os.path.join(Data_folder_path,
                                                                                                           'testing_set.pkl')
        print(f'Work on {args.set_name}!')

    with open(set_path, 'rb') as file:
        records = pickle.load(file)

    # Create result folder
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder = f"results_{current_time}"
    os.makedirs(result_folder, exist_ok=True)
    folder_path = os.path.abspath(result_folder)

    # Information Extraction
    client = OpenAI()
    for record in records:
        # Prepare record
        record_path = os.path.join(args.data_path,record)
        # Encode crash sketch
        sketch = encode_image(os.path.join(record_path, 'Sketch.jpg'))
        # Get crash summary
        with open(os.path.join(record_path, 'Summary.txt'), 'r', encoding='utf-8') as file:
            summary = file.read()

        # Prepare prompts & resources for prompt engineering
        road_type_sketch = encode_image(r'./prompts/Sketch.jpg')  # Curve example
        with open(
                r'./prompts/Summary.txt',
                'r', encoding='utf-8') as file:
            road_type_summary = file.read()
        with open(
                r'./prompts/p1.txt',
                'r', encoding='utf-8') as file:
            user_prompts_1 = file.read()
        with open(
                r'./prompts/p2.txt',
                'r', encoding='utf-8') as file:
            user_prompts_2 = file.read()

        road_type_extraction(sketch,
                             summary,
                             road_type_sketch,
                             road_type_summary,
                             user_prompts_1,
                             user_prompts_2,
                             client,
                             record,
                             folder_path,
                             args.gpt)
        print(f"Case {record} finished!")
        time.sleep(1)

    # Save extraction results
    pre_results = read_txt_results(folder_path)
    df = pd.DataFrame({
            'ID': list(pre_results.keys()),
            'Label': [value[0] for value in pre_results.values()]
        })
    file_path = os.path.join(folder_path, 'road_type.xlsx')
    df.to_excel(file_path, index=False)

    # Save results
    with open(os.path.join(folder_path, 'road_type_results.pkl'), 'wb') as f:
        pickle.dump(pre_results, f)
    print(f"Results have been saved!")


if __name__ == '__main__':
    main()
