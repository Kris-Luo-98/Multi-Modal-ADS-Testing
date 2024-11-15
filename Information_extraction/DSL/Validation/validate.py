import argparse
import os
from datetime import datetime
import yaml
import pickle

def check_actor(true_DSL,prediction_DSL):
    # 0 for failed validation
    # 1 for passed validation
    flag = 1
    value = prediction_DSL['Actors'].pop('Validation', None)
    if len(true_DSL['Actors']) != len(prediction_DSL['Actors'])/2:
        return 0
    else:
        for key, value in true_DSL['Actors'].items():
            if value != prediction_DSL['Actors'][key]:
                flag = 0
    return flag

def check_env(true_DSL,prediction_DSL):
    flag = 1

    value = prediction_DSL['Env'].pop('Validation', None)
    for key, value in true_DSL['Env'].items():
        if value != prediction_DSL['Env'][key]:
            flag = 0
    return flag

def check_road(true_DSL,prediction_DSL):
    len_thresh = 10
    # road type
    value = prediction_DSL['Road network'].pop('Validation', None)
    if true_DSL['Road type'] != prediction_DSL['Road type']:
        return 0
    elif len(true_DSL['Road network']) != len(prediction_DSL['Road network']):
        return 0
    else:
        # check length & width
        if true_DSL['Road type'] in ['Curve','Intersection','Straight']:
            # check length & width
            if abs(true_DSL['Road network']['Length'] - prediction_DSL['Road network']['Length']) > len_thresh:
                return 0
            if abs(true_DSL['Road network']['Width'] - prediction_DSL['Road network']['Width']) > 1:
                return 0
            if true_DSL['Road network']['No_lanes'] != prediction_DSL['Road network']['No_lanes']:
                return 0
            if true_DSL['Road network']['No_ways'] != prediction_DSL['Road network']['No_ways']:
                return 0
        elif true_DSL['Road type'] == 'T-intersection':
            if abs(true_DSL['Road network']['Length_branch'] - prediction_DSL['Road network']['Length_branch']) > len_thresh:
                return 0
            if abs(true_DSL['Road network']['Length_main'] - prediction_DSL['Road network']['Length_main']) > len_thresh:
                return 0
            if abs(true_DSL['Road network']['Width'] - prediction_DSL['Road network']['Width']) > 1:
                return 0
            if true_DSL['Road network']['No_lanes_branch_road'] != prediction_DSL['Road network']['No_lanes_branch_road']:
                return 0
            if true_DSL['Road network']['No_lanes_main_road'] != prediction_DSL['Road network']['No_lanes_main_road']:
                return 0
            if true_DSL['Road network']['No_ways_branch_road'] != prediction_DSL['Road network']['No_ways_branch_road']:
                return 0
            if true_DSL['Road network']['No_ways_main_road'] != prediction_DSL['Road network']['No_ways_main_road']:
                return 0
        elif true_DSL['Road type'] == 'Merge':
            if abs(true_DSL['Road network']['Length_main'] - prediction_DSL['Road network']['Length_main']) > len_thresh:
                return 0
            if abs(true_DSL['Road network']['Width'] - prediction_DSL['Road network']['Width']) > 1:
                return 0
            if true_DSL['Road network']['No_lanes_branch_road'] != prediction_DSL['Road network']['No_lanes_branch_road']:
                return 0
            if true_DSL['Road network']['No_lanes_main_road'] != prediction_DSL['Road network']['No_lanes_main_road']:
                return 0
            if true_DSL['Road network']['No_ways_main_road'] != prediction_DSL['Road network']['No_ways_main_road']:
                return 0
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        default=r'C:\Users\Kris\Desktop\Multi-Modal-ADS-Testing\Information_extraction\DSL\Validation\Golden_Oracle_by_human_validator_1',
                        type=str)
    parser.add_argument('--prediction',
                        default=r'C:\Users\Kris\Desktop\Multi-Modal-ADS-Testing\Information_extraction\DSL\Encoded_2024-11-04_00-13-48',
                        type=str)
    args = parser.parse_args()

    # True targe DSL_DSL
    dsl_files = {}
    for file_name in os.listdir(args.data_path):
        case_id = file_name.split('.')[0]
        dsl_files[case_id] = os.path.join(args.data_path, file_name)

    num_records = len(dsl_files)
    actor_pass = 0
    env_pass = 0
    road_pass = 0
    all_pass = 0

    # evaluation metrics
    # accuracy: All fields are considered accurate if they match
    # env_accuracy
    # road_accuracy
    # actor_accuracy

    for key, value in dsl_files.items():
        # True labels
        with open(value, 'r', encoding='utf-8') as file:
            true_DSL = yaml.safe_load(file)
        # Prediction results
        with open(os.path.join(args.prediction, f"{key}.yaml"), 'r', encoding='utf-8') as file:
            prediction_DSL = yaml.safe_load(file)

        ac = check_actor(true_DSL,prediction_DSL)
        actor_pass += ac
        en = check_env(true_DSL,prediction_DSL)
        env_pass += en
        ro = check_road(true_DSL,prediction_DSL)
        road_pass += ro
        if ac == en == ro == 1:
            all_pass += 1
        # print summary
        print('*********************************************')
        print(f"Validation result for record - {key}: \n"
              f"Actor part: {ac}\n"
              f"Env part: {en}\n"
              f"Road part: {ro}\n")

    actor_accuracy = actor_pass / num_records
    env_accuracy = env_pass / num_records
    road_accuracy = road_pass / num_records
    accuracy = all_pass / num_records

    print(f"Actor Accuracy: {actor_accuracy}")
    print(f"Env Accuracy: {env_accuracy}")
    print(f"Road Accuracy: {road_accuracy}")
    print(f"Overall Accuracy: {accuracy}")

    accuracy_results = {
        'actor_accuracy': actor_accuracy,
        'env_accuracy': env_accuracy,
        'road_accuracy': road_accuracy,
        'overall_accuracy': accuracy
    }
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkl_file_path = f'accuracy_results_{current_time}.pkl'
    with open(pkl_file_path, 'wb') as file:
        pickle.dump(accuracy_results, file)


if __name__=='__main__':
    main()