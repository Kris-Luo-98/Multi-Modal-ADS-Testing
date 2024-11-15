import pandas as pd
import pickle
import os
import yaml
from datetime import datetime
import argparse

def check_dict_lengths_equal(*dicts):
    lengths = [len(d) for d in dicts]
    return len(set(lengths)) == 1


def main():
    IE_path = os.path.dirname(os.getcwd())
    parser = argparse.ArgumentParser(description='MM ADS Testing - road type extraction')
    parser.add_argument('--road_type', default=os.path.join(IE_path, 'Road_type',
                                             'results_2024-10-30_21-26-59', 'road_type_results.pkl'))
    parser.add_argument('--road_network', default=os.path.join(IE_path, 'Road_network',
                                             'results_2024-10-30_21-58-04', 'road_network_results.pkl'))
    parser.add_argument('--actros', default=os.path.join(IE_path, 'Actors',
                                             'results_2024-11-03_18-31-50', 'actor_results.pkl'))
    parser.add_argument('--env_info', default=os.path.join(IE_path, 'Env',
                                             'results_2024-10-30_23-27-34', 'env_info.pkl'))
    args = parser.parse_args()

    with open(args.road_type, 'rb') as f:
        pre_results = pickle.load(f)
    id_label_dict = {ID: values[0] for ID, values in pre_results.items()}

    # Road network
    with open(args.road_network, 'rb') as f:
        road_network_dict = pickle.load(f)

    # Traj
    with open(args.actros, 'rb') as f:
        traj_dict = pickle.load(f)

    # Env info
    with open(args.env_info, 'rb') as f:
        env_info = pickle.load(f)

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder = f"Encoded_{current_time}"
    os.makedirs(result_folder, exist_ok=True)
    folder_path = os.path.abspath(result_folder)

    are_lengths_equal = check_dict_lengths_equal(id_label_dict, road_network_dict, traj_dict,
                                                 env_info)
    # are_lengths_equal = True
    if are_lengths_equal:
        for id_, label in id_label_dict.items():
            id_ = str(id_)
            encoded_data = {
                'Scenario': id_,
                'Road type': label,
                'Road network': road_network_dict[id_],
                'Actors': traj_dict[id_],
                'Env': env_info[id_]
            }
            file_name = os.path.join(folder_path, f'{id_}.yaml')
            with open(file_name, 'w') as yaml_file:
                yaml.dump(encoded_data, yaml_file, default_flow_style=False, allow_unicode=True)


if __name__=='__main__':
    main()