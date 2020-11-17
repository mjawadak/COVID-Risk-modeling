import pandas as pd
import numpy as np
from getRiskIndividual import get_geo_risk


# Provide the scenario IDs here. Obtained after running the simulation (run_simulation.py)
scenario_filenames=["OUTPUT_110011_20201117123025"]



def consolidate_risk_per_location(simulation_file):
    FOLDER = "simulation_output/"
    simulation_file_main = FOLDER + simulation_file + "_overall_agent_status.csv"
    dfs = pd.read_csv(simulation_file_main)
    dfs["date"] = pd.to_datetime(dfs.loc[:, "Date_Time"]).dt.date

    dfs_g = dfs.groupby(["Date_Time", "currentLocationID", "currentLocationType"]).mean()
    dfs_g2 = dfs.groupby(["Date_Time", "currentLocationID", "currentLocationType"]).count()
    dfs_g3 = dfs.query("infection_status == 'infected'").groupby(
        ["Date_Time", "currentLocationID", "currentLocationType"]).count()

    dfs_g = dfs_g.reset_index()
    dfs_g2 = dfs_g2.reset_index()
    dfs_g3 = dfs_g3.reset_index()

    dfs_all = dfs_g.merge(dfs_g2, how="left", on=["Date_Time", "currentLocationID", "currentLocationType"])
    dfs_all = dfs_all.merge(dfs_g3, how="left", on=["Date_Time", "currentLocationID", "currentLocationType"])
    dfs_all = dfs_all.fillna(0)
    dfs_all_final = dfs_all.loc[:,["Date_Time", "currentLocationID", "currentLocationType", "zone_id_x", "lat_x", "lon_x", "age_x",
                                   "gender_x", "days_since_infection_x", "infection_status_x", "infection_status_y"]]

    dfs_all_final = dfs_all_final.rename(
        columns={"age_x": "avg_age", "lat_x": "lat", "lon_x": "lon", "zone_id_x": "zone_id",
                 "gender_x": "avg_gender",
                 "days_since_infection_x": "avg_days_since_infection",
                 "infection_status_x": "proximity",
                 "infection_status_y": "proximity_infected"
                 })

    risks = []
    for row in dfs_all.loc[:,["Date_Time",
                              "currentLocationID",
                              "age_x",
                              "gender_x",
                              "days_since_infection_x",
                              "infection_status_x","infection_status_y"]].values:
        risk,age_factor,gender_factor,proximity_factor,infected_individuals_factor,infectiousness_factor = get_geo_risk(row[2],row[3],row[4],row[5],row[6])
        risks.append(risk)
    dfs_all_final["risk"] = risks

    dfs_all_final["date"]=pd.to_datetime(dfs_all_final.loc[:,"Date_Time"]).dt.date
    dfs_all_final_gp = dfs_all_final.groupby(["date","currentLocationID"]).max().reset_index()
    dfs_all_final_gp["date"] = pd.to_datetime(dfs_all_final_gp["date"])

    dfs_all_final_gp.to_csv(FOLDER+simulation_file+"_overall_geo_status.csv")

if __name__ == "__main__":
    for i in range(len(scenario_filenames)):

        consolidate_risk_per_location(scenario_filenames[i])

        print(i)