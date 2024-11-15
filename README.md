# Multi-Modal-ADS-Testing

Dataset -> 50 records (combining LCTGen and ADEPT), saved under Crash_dataset

Experiments:

1. RQ 1 DSL validation

    Make 2 golden oracle (including ENV, road network and vehicle type info), saved under Information_extraction -> DSL -> Validation.
    T-test to check the consistency between these 2 golden oracle.
    One is already done by Siwei. Saved in Golden_oracle_1.
    Another one will be done by Fida.

    run validate.py to report DSL accuracy on the ENV, Roadnetwork and Vehicle type info. -> Siwei, Luo.

    run traj_vis.py to visualize the trajectory, and design human study to check if the trajectories are consistent with the ones in crash sketch. scale 1-5, 1 is total not agree, 5 is pretty agree. T-test -> Yang

2. RQ 2 Scenario validation

    Generate scenarios on MetaDrive with ADS IDM Policy -> Siwei (output:how long, scenario numbers, crash numbers, recordings)

    Generate scenarios on MetaDrive with ADS PPO Policy -> Siwei (output:how long, scenario numbers, crash numbers, recordings)

    Generate scenarios on BeamNG with Auto -> Siwei (output:how long, scenario numbers, crash numbers, recordings)

    Do a humany study, check whether the scenario are real and consistent with the crash kappa test -> Yang

3. RQ 3 Utility

    Total Number of Bugs found in the whole experiments -> Siwei

    Generation time -> Siwei

    In one simulator, to find the top-k (3, 5) bugs, how many scenarios do i need. -> Siwei

    In one simulator, 50 scenarios can find how many bugs. -> Siwei

4. RQ 4 Ablitation study

    Remove the validation procress in KE, how's the performance -> Siwei

    Combine RT + RN, how's the performance. -> Siwei
