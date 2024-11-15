from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.vehicle.vehicle_type import DefaultVehicle, XLVehicle
from metadrive.utils import generate_gif
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.scenario.parse_object_state import get_idm_route
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
import argparse
import os
import yaml
import time
import math
import numpy as np
from datetime import datetime
from metadrive.engine.logger import set_log_level
import logging
import copy

def convert_string_to_list_of_lists(string):
    string = string.strip('[]')
    list_of_tuples = eval(string)
    list_of_lists = [list(item) for item in list_of_tuples]
    return list_of_lists

def get_sim_time(env_info):
    day_time_mapping = {
        'Daytime': '11:00',
        'Nighttime': '20:00'
    }
    return day_time_mapping.get(env_info.get('Time'), '11:00')

def get_car_info(actors):
    if 'Validation' in actors:
        no_cars = int((len(actors) - 1) / 2)
    else:
        no_cars = int(len(actors) / 2)

    car_dict = {}
    for i in range(no_cars):
        car_dict[f"V{i + 1}"] = [
            convert_string_to_list_of_lists(actors[f"V{i + 1}_traj"]),
            actors[f"V{i + 1}_type"]
        ]
    return car_dict

def load_yaml_files_to_dict(directory_path):
    yaml_dict = {}

    for filename in os.listdir(directory_path):
        if filename.endswith(".yaml"):
            file_id = os.path.splitext(filename)[0]
            file_path = os.path.join(directory_path, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                content = yaml.safe_load(file)
                yaml_dict[file_id] = content

    return yaml_dict

def leftorright(waypoints):
    p1_x = waypoints[0][0]
    p2_x = waypoints[1][0]
    if p2_x - p1_x >= 0:
        return 'right'
    else:
        return 'left'

def way2nodes_S(car_dict,road_network,lane_num):
    nodes = {}
    # Place Ego car first
    ego_nodes = []
    waypoints_ego = car_dict['V1'][0]
    heading = leftorright(waypoints_ego)
    if heading == 'right':
        ego_nodes.append('>>')
        ego_nodes.append('>>>')
        if car_dict['V1'][0][-1][-1] >= (lane_num * road_network['Width']):
            ego_nodes.append('-1S0_0_')
        else:
            ego_nodes.append('1S0_0_')
    else:
        ego_nodes.append('->>>')
        ego_nodes.append('->>')
        if car_dict['V1'][0][-1][-1] >= (lane_num * road_network['Width']):
            ego_nodes.append('->')
        else:
            ego_nodes.append('>')
    ego_nodes.append(min(int(abs(waypoints_ego[0][1]) // road_network['Width']), lane_num -1))
    nodes['Ego'] = ego_nodes
    # Place NPC 1
    npc_nodes = []
    waypoints_npc = car_dict['V2'][0]
    heading = leftorright(waypoints_npc)
    if heading == 'right':
        # before ego or after?
        npc_x = waypoints_npc[0][0]
        ego_x = waypoints_ego[0][0]
        if npc_x >= ego_x:
            npc_nodes.append('>>>')
            npc_nodes.append('1S0_0_')
        else:
            npc_nodes.append('>')
            npc_nodes.append('>>')
        if car_dict['V2'][0][-1][-1] >= (lane_num * road_network['Width']):
            npc_nodes.append('-1S0_0_')
        else:
            npc_nodes.append('1S0_0_')
    else:
        # before ego or after?
        npc_x = waypoints_npc[0][0]
        ego_x = waypoints_ego[0][0]
        if npc_x >= ego_x:
            npc_nodes.append('-1S0_0_')
            npc_nodes.append('->>>')
        else:
            npc_nodes.append('->>')
            npc_nodes.append('->')
        if car_dict['V2'][0][-1][-1] >= (lane_num * road_network['Width']):
            npc_nodes.append('->')
        else:
            npc_nodes.append('>')
    npc_nodes.append(min(int(abs(waypoints_ego[0][1]) // road_network['Width']), lane_num -1))
    nodes['NPC'] = npc_nodes
    # Place NPC 2
    if len(car_dict) == 3:
        npc_nodes = []
        waypoints_npc = car_dict['V3'][0]
        heading = leftorright(waypoints_npc)
        if heading == 'right':
            # before ego or after?
            npc_x = waypoints_npc[0][0]
            ego_x = waypoints_ego[0][0]
            if npc_x >= ego_x:
                npc_nodes.append('>>>')
                npc_nodes.append('1S0_0_')
            else:
                npc_nodes.append('>')
                npc_nodes.append('>>')
            if car_dict['V3'][0][-1][-1] >= (lane_num * road_network['Width']):
                npc_nodes.append('-1S0_0_')
            else:
                npc_nodes.append('1S0_0_')
        else:
            # before ego or after?
            npc_x = waypoints_npc[0][0]
            ego_x = waypoints_ego[0][0]
            if npc_x >= ego_x:
                npc_nodes.append('-1S0_0_')
                npc_nodes.append('->>>')
            else:
                npc_nodes.append('->>')
                npc_nodes.append('->')
            if car_dict['V2'][0][-1][-1] >= (lane_num * road_network['Width']):
                npc_nodes.append('->')
            else:
                npc_nodes.append('>')
        npc_nodes.append(min(int(abs(waypoints_ego[0][1]) // road_network['Width']), lane_num -1))
        nodes['NPC_2'] = npc_nodes
    return nodes

def LRUD(waypoints):
    p1_x = waypoints[0][0]
    p1_y = waypoints[0][1]
    pn_x = waypoints[-1][0]
    pn_y = waypoints[-1][1]
    x_d = pn_x - p1_x
    y_d = pn_y - p1_y
    if abs(x_d) >= abs(y_d):
        # go along X axis
        if x_d >= 0:
            return 'right'
        else:
            return 'left'
    else:
        # go along Y axis
        if y_d >= 0:
            return 'up'
        else:
            return 'down'

def extract_position(car,car_dict,road_network,lane_num):
    ego_nodes = []
    waypoints_ego = car_dict[car][0]
    heading = LRUD(waypoints_ego)
    if heading == 'right':
        # which handside?
        if waypoints_ego[0][0] <= 0:
            # >> -> >>>
            ego_nodes.append('>>')
            ego_nodes.append('>>>')
            # dest
            go_up_down = abs(waypoints_ego[-1][1])
            go_straight = abs(waypoints_ego[-1][0])
            if go_up_down >= go_straight:
                if waypoints_ego[-1][1] >= 0:
                    # set dest
                    ego_nodes.append('1X2_0_')
                else:
                    ego_nodes.append('1X0_0_')
            else:
                ego_nodes.append('1X1_0_')
            # lane
            ego_nodes.append(min(int(abs(waypoints_ego[0][1]) // road_network['Width']), lane_num - 1))
        else:
            # >>> -> 1x1_0_
            ego_nodes.append('>>>')
            ego_nodes.append('1X1_0_')
            # dest
            if waypoints_ego[-1][1] >= 0:
                # set dest
                ego_nodes.append('-1X1_1_')
            else:
                ego_nodes.append('1X1_1_')
            # lane
            ego_nodes.append(min(int(abs(waypoints_ego[0][1]) // road_network['Width']), lane_num - 1))
    elif heading == 'left':
        # which handside?
        if waypoints_ego[0][0] > 0:
            # -1x1_1_ -> -1x1_0_
            ego_nodes.append('-1X1_1_')
            ego_nodes.append('-1X1_0_')
            # dest
            go_up_down = abs(waypoints_ego[-1][1])
            go_straight = abs(waypoints_ego[-1][0])
            if go_up_down >= go_straight:
                if waypoints_ego[-1][1] >= 0:
                    # set dest
                    ego_nodes.append('1X2_0_')
                else:
                    ego_nodes.append('1X0_0_')
            else:
                ego_nodes.append('->>>')
            # lane
            ego_nodes.append(min(int(abs(waypoints_ego[0][1]) // road_network['Width']), lane_num - 1))
        else:
            # -1x1_0_ -> ->>>
            ego_nodes.append('-1X1_0_')
            ego_nodes.append('->>>')
            # dest
            if waypoints_ego[-1][1] >= 0:
                # set dest
                ego_nodes.append('->>')
            else:
                ego_nodes.append('>>')
            # lane
            ego_nodes.append(min(int(abs(waypoints_ego[0][1]) // road_network['Width']), lane_num - 1))
    elif heading == 'up':
        # which handside?
        if waypoints_ego[0][1] <= 0:
            # -1x0_1_ -> -1x0_0_
            ego_nodes.append('-1X0_1_')
            ego_nodes.append('-1X0_0_')
            # dest
            go_left_right = abs(waypoints_ego[-1][0])
            go_straight = abs(waypoints_ego[-1][1])
            if go_left_right >= go_straight:
                if waypoints_ego[-1][0] >= 0:
                    # set dest
                    ego_nodes.append('1X1_0_')
                else:
                    ego_nodes.append('->>>')
            else:
                ego_nodes.append('1X2_0_')
            # lane
            ego_nodes.append(min(int(abs(waypoints_ego[0][0]) // road_network['Width']), lane_num - 1))
        else:
            # -1x0_0_ -> 1x2_0_
            ego_nodes.append('-1X0_0_')
            ego_nodes.append('1X2_0_')
            # dest
            if waypoints_ego[-1][0] >= 0:
                # set dest
                ego_nodes.append('1X2_1_')
            else:
                ego_nodes.append('-1X2_1_')
            # lane
            ego_nodes.append(min(int(abs(waypoints_ego[0][0]) // road_network['Width']), lane_num - 1))
    elif heading == 'down':
        # which handside?
        if waypoints_ego[0][1] > 0:
            # -1x2_1_ -> -1x2_0_
            ego_nodes.append('-1X2_1_')
            ego_nodes.append('-1X2_0_')
            # dest
            go_left_right = abs(waypoints_ego[-1][0])
            go_straight = abs(waypoints_ego[-1][1])
            if go_left_right >= go_straight:
                if waypoints_ego[-1][0] >= 0:
                    # set dest
                    ego_nodes.append('1X1_0_')
                else:
                    ego_nodes.append('->>>')
            else:
                ego_nodes.append('1X0_0_')
            # lane
            ego_nodes.append(min(int(abs(waypoints_ego[0][0]) // road_network['Width']), lane_num - 1))
        else:
            # -1x2_0_ -> 1x0_0_
            ego_nodes.append('-1X2_0_')
            ego_nodes.append('1X0_0_')
            # dest
            if waypoints_ego[-1][0] >= 0:
                # set dest
                ego_nodes.append('-1X0_1_')
            else:
                ego_nodes.append('1X0_1_')
            # lane
            ego_nodes.append(min(int(abs(waypoints_ego[0][0]) // road_network['Width']), lane_num - 1))
    return ego_nodes

def way2nodes_X(car_dict,road_network,lane_num):
    nodes = {}
    nodes['Ego'] = extract_position('V1',car_dict,road_network,lane_num)
    nodes['NPC'] = extract_position('V2',car_dict,road_network,lane_num)
    if len(car_dict) == 3:
        nodes['NPC_2'] = extract_position('V3',car_dict,road_network,lane_num)
    return nodes

def extract_position_t(car,car_dict,road_network,lane_num):
    ego_nodes = []
    waypoints_ego = car_dict[car][0]
    heading = LRUD(waypoints_ego)
    if heading == 'right':
        # which handside?
        if waypoints_ego[0][0] <= 0:
            # >> -> >>>
            ego_nodes.append('>>')
            ego_nodes.append('>>>')
            # dest
            go_down = abs(waypoints_ego[-1][1])
            go_straight = abs(waypoints_ego[-1][0])
            if go_down >= go_straight:
                    ego_nodes.append('1T0_0_')
            else:
                ego_nodes.append('1T1_0_')
            # lane
            ego_nodes.append(min(int(abs(waypoints_ego[0][1]) // road_network['Width']), lane_num - 1))
        else:
            # >>> -> 1T1_0_
            ego_nodes.append('>>>')
            ego_nodes.append('1T1_0_')
            # dest
            if waypoints_ego[-1][1] >= 0:
                # set dest
                ego_nodes.append('-1T1_1_')
            else:
                ego_nodes.append('1T1_1_')
            # lane
            ego_nodes.append(min(int(abs(waypoints_ego[0][1]) // road_network['Width']), lane_num - 1))
    elif heading == 'left':
        # which handside?
        if waypoints_ego[0][0] > 0:
            # -1x1_1_ -> -1x1_0_
            ego_nodes.append('-1T1_1_')
            ego_nodes.append('-1T1_0_')
            # dest
            go_down = abs(waypoints_ego[-1][1])
            go_straight = abs(waypoints_ego[-1][0])
            if go_down >= go_straight:
                    ego_nodes.append('1T0_0_')
            else:
                ego_nodes.append('->>>')
            # lane
            ego_nodes.append(min(int(abs(waypoints_ego[0][1]) // road_network['Width']), lane_num - 1))
        else:
            # -1T1_0_ -> ->>>
            ego_nodes.append('-1T1_0_')
            ego_nodes.append('->>>')
            # dest
            if waypoints_ego[-1][1] >= 0:
                # set dest
                ego_nodes.append('->>')
            else:
                ego_nodes.append('>>')
            # lane
            ego_nodes.append(min(int(abs(waypoints_ego[0][1]) // road_network['Width']), lane_num - 1))
    elif heading == 'up':
        # which handside?
        if waypoints_ego[-1][0] >= 0:
            ego_nodes.append('-1T0_0_')
            ego_nodes.append('1T1_0_')
            ego_nodes.append('-1T1_1_')
        else:
            ego_nodes.append('-1T0_0_')
            ego_nodes.append('->>>')
            ego_nodes.append('>>')
        # lane
        ego_nodes.append(min(int(abs(waypoints_ego[0][1]) // road_network['Width']), lane_num - 1))
    return ego_nodes

def way2nodes_T(car_dict,road_network,lane_num):
    nodes = {}
    nodes['Ego'] = extract_position_t('V1', car_dict, road_network, lane_num)
    nodes['NPC'] = extract_position_t('V2', car_dict, road_network, lane_num)
    if len(car_dict) == 3:
        nodes['NPC_2'] = extract_position_t('V3', car_dict, road_network, lane_num)
    return nodes

def way2nodes_C(car_dict,road_network,lane_num):
    nodes = {}
    ego_traj = ['>>', '>>>']
    if (car_dict['V1'][0][-1][0] - car_dict['V2'][0][-1][0]) < 5:
        ego_traj.append('-1C0_0_')
    else:
        ego_traj.append('1C0_0_')
    ego_traj.append(0)
    nodes['Ego'] = ego_traj
    nodes['NPC'] = ['-1C0_1_','-1C0_0_','>>>',0]
    return nodes

def way2nodes_R(car_dict,road_network,lane_num):
    nodes = {}
    # Ego car
    ego_traj = []
    # whether straight or going up
    if (car_dict['V1'][0][-1][1] - car_dict['V1'][0][0][1]) > 7:
        # go up
        ego_traj.append('1r1_1_')
        ego_traj.append('1r1_2_')
        ego_traj.append('-1r0_2_')
        ego_traj.append(0)
        nodes['Ego'] = ego_traj
        nodes['NPC'] = ['>>', '>>>', '1r0_1_', 0]
    else:
        # go straight
        if (car_dict['V1'][0][1][0] - car_dict['V1'][0][0][0]) >= 0:
            # go right
            ego_traj.append('>>')
            ego_traj.append('>>>')
            ego_traj.append('-1r0_2_')
            ego_traj.append(0)
            nodes['Ego'] = ego_traj
        else:
            ego_traj.append('-1r0_1_')
            ego_traj.append('-1r0_0_')
            ego_traj.append('>>>')
            ego_traj.append(0)
            nodes['Ego'] = ego_traj
        nodes['NPC'] = ['>>', '>>>', '1r0_1_', 0]
    return nodes


def calculate_distance(point1, point2):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def static_npc(car_dict):
    static_list = []
    # Select from the second car
    # the 1st has been set as ego car
    for i in range(1, len(car_dict)):
        idx = i + 1
        traj = car_dict[f"V{idx}"][0]
        # distance
        dis = calculate_distance(traj[0],traj[-1])
        if dis < 3:
            static_list.append(1)
        else:
            static_list.append(0)
    return static_list

def run_scenario(road_network, env_info, actors, road_type, ID, folder_path, seq):
    crash_flag = 0
    print('=================')
    print(f'Case: {ID}')
    print('-----')
    set_log_level(logging.CRITICAL)
    day_time = get_sim_time(env_info)
    car_dict = get_car_info(actors)
    # car_dict example
    # {'V1': [[[4, 2], [10, 3], [15, 4], [20, 5], [25, 6]], 'Car'], 'V2': [[[10, 6], [13, 6], [18, 7], [21, 6], [25, 6]], 'Truck']}
    if car_dict['V1'][1] == 'Car':
        ego_model = 'default'
    else:
        ego_model = 'xl'
    # Map construction
    if road_type == 'Straight':
        s_lane_num = int((road_network['No_lanes'] + 1) / 2) if road_network['No_lanes'] % 2 != 0 else int(
            road_network['No_lanes'] / 2)
        car_config = way2nodes_S(car_dict, road_network, s_lane_num)
        map_con = 'S'
    elif road_type == 'Intersection':
        s_lane_num = int((road_network['No_lanes'] + 1) / 2) if road_network['No_lanes'] % 2 != 0 else int(
            road_network['No_lanes'] / 2)
        car_config = way2nodes_X(car_dict, road_network, s_lane_num)
        map_con = 'X'
    elif road_type == 'T-intersection':
        s_lane_num = int((road_network['No_lanes_main_road'] + 1) / 2) if road_network['No_lanes_main_road'] % 2 != 0 else int(
            road_network['No_lanes'] / 2)
        car_config = way2nodes_T(car_dict, road_network, s_lane_num)
        map_con = 'T'
    elif road_type == 'Curve':
        s_lane_num = int((road_network['No_lanes'] + 1) / 2) if road_network['No_lanes'] % 2 != 0 else int(
            road_network['No_lanes'] / 2)
        car_config = way2nodes_C(car_dict, road_network, s_lane_num)
        map_con = 'C'
    elif road_type == 'Merge':
        # s_lane_num = int((road_network['No_lanes'] + 1) / 2) if road_network['No_lanes'] % 2 != 0 else int(
        #     road_network['No_lanes'] / 2)
        s_lane_num = 4
        car_config = way2nodes_R(car_dict, road_network, s_lane_num)
        map_con = 'r'
    print(car_config)
    print('-----')
    print(road_network)
    print('-----')
    print(actors)
    scenario_config = {'map_config': {'type': 'block_sequence',
                                      'config': map_con,
                                      'lane_num': s_lane_num,
                                      },
                       'agent_policy': ExpertPolicy,
                       'traffic_density': 0,
                       'agent_configs': {
                           'default_agent': {
                               'use_special_color': True,
                               'spawn_lane_index': (car_config['Ego'][0], car_config['Ego'][1], car_config['Ego'][3]),
                               # Extract from DSL
                               'destination': car_config['Ego'][2],  # Extract from DSL
                               'vehicle_model': ego_model  # Not sure if it is possible
                           }
                       },
                       'use_render': True,
                       'daytime': day_time,
                       }
    env = MetaDriveEnv(scenario_config)
    frames = []
    # check whether npc is static
    # Objects moving less than 5 meters are considered stationary.
    static_list = static_npc(car_dict)
    try:
        env.reset()
        # Build NPC 1
        cfg = copy.deepcopy(env.config["vehicle_config"])
        cfg["navigation_module"] = NodeNetworkNavigation
        cfg['spawn_lane_index'] = (car_config['NPC'][0], car_config['NPC'][1], car_config['NPC'][3])  # Extract from DSL
        cfg['destination'] = car_config['NPC'][2]  # Extract from DSL
        if car_dict['V2'][1] == 'Car':
            npc_model = DefaultVehicle
        else:
            npc_model = XLVehicle
        npc = env.engine.spawn_object(npc_model, vehicle_config=cfg)
        env.engine.add_policy(npc.id, ExpertPolicy, npc, env.engine.generate_seed())
        # Build NPC 2
        if len(car_dict) == 3:
            cfg_1 = copy.deepcopy(env.config["vehicle_config"])
            cfg_1["navigation_module"] = NodeNetworkNavigation
            cfg_1['spawn_lane_index'] = (
                car_config['NPC_2'][0], car_config['NPC_2'][1], car_config['NPC_2'][3])  # Extract from DSL
            cfg_1['destination'] = car_config['NPC_2'][2]  # Extract from DSL
            if car_dict['V3'][1] == 'Car':
                npc_model_1 = DefaultVehicle
            else:
                npc_model_1 = XLVehicle
            npc_1 = env.engine.spawn_object(npc_model_1, vehicle_config=cfg_1)
            env.engine.add_policy(npc_1.id, ExpertPolicy, npc_1, env.engine.generate_seed())

        for _ in range(100):
            # NPC action
            if static_list[0] == 0:
                npc.before_step(env.engine.get_policy(npc.name).act(True))
            if len(car_dict) == 3:
                if static_list[1] == 0:
                    npc_1.before_step(env.engine.get_policy(npc_1.name).act(True))
            p = env.engine.get_policy(env.agent.name)
            _, r, _, _, info = env.step(p.act(True))

            # frame = env.render(mode="topdown",
            #                    window=False,
            #                    film_size=(2000, 200),
            #                    screen_size=(550, 200),
            #                    draw_target_vehicle_trajectory=False,
            #                    scaling=10,
            #                    camera_position=None)
            frame = env.render(mode="topdown",
                               window=False,
                               # film_size=(2000, 200),
                               screen_size=(800, 400),
                               draw_target_vehicle_trajectory=False,
                               scaling=4,
                               camera_position=None)

            frames.append(frame)
            if info['crash']:
                crash_flag = 1
                break
        name_gif = folder_path + f"\{ID}_{seq}.gif"
        generate_gif(frames, gif_name=name_gif)
    finally:
        env.close()
    return crash_flag
def main():
    parser = argparse.ArgumentParser(description='MM ADS Testing - Scenario Reconstruction in MetaDrive')
    parser.add_argument('--dsl_path', default=r'C:\Users\Kris\Desktop\Multi-Modal-ADS-Testing\Information_extraction\DSL\Encoded_2024-11-04_00-13-48', type=str)
    args = parser.parse_args()

    result_folder = f".\\PPOPolicy\\results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(result_folder, exist_ok=True)
    folder_path = os.path.abspath(result_folder)

    yaml_data = load_yaml_files_to_dict(args.dsl_path)
    fail_list = []
    num_cases = len(yaml_data)
    ss_count = 0
    crash_count = 0

    for ID, DSL in yaml_data.items():
        actors = DSL['Actors']
        env_info = DSL['Env']
        road_network = DSL['Road network']
        road_type = DSL['Road type']
        for i in [1, 2]:
            if i == 1:
                # construct it at the first time
                try:
                    crash_sta = run_scenario(road_network, env_info, actors, road_type, ID, folder_path, i)
                    ss_count += 1
                    crash_count += crash_sta
                except Exception as e:
                    print(f"Scenario {ID} build failed due to error: {e}")
                    fail_list.append(ID)
                # end -> swap vehicles
            elif i == 2:
                try:
                    # swap vehicles
                    V1_type = actors['V1_type']
                    V1_traj = actors['V1_traj']
                    V2_type = actors['V2_type']
                    V2_traj = actors['V2_traj']
                    actors['V1_type'] = V2_type
                    actors['V1_traj'] = V2_traj
                    actors['V2_type'] = V1_type
                    actors['V2_traj'] = V1_traj
                    crash_sta = run_scenario(road_network, env_info, actors, road_type, ID, folder_path, i)
                    ss_count += 1
                    crash_count += crash_sta
                except Exception as e:
                    print(f"Scenario {ID} build failed due to error: {e}")
                    fail_list.append(ID)

        print('============= Simulation Results =============')
        print('Total Number of Cases: ', num_cases)
        print(f'Successfully built {ss_count} examples!')
        print(f'{crash_count} collision cases have been found!')
        print('==============================================')


if __name__=='__main__':
    main()