import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



scenario_filenames = ["OUTPUT_110011_20201117123025"]

scenario_labels =["Lockdown enabled,Self Isolation,Mask Compliance (0.5)"]


MAX_DAY = 250#250#120
POPULATION = 10000.0
FIGSIZE = [20,10]
plt.rcParams.update({'font.size': 22})


#### comparison of infections
plt.figure(figsize=FIGSIZE)
for i in range(len(scenario_labels)):
    if True:#i in [1,3,4]:
        simulation_file = "simulation_output/"+scenario_filenames[i]+".csv"
        df = pd.read_csv(simulation_file)
        dfg = df.groupby("Date").mean()
        last_val = (100*dfg["Infected_count"].values/POPULATION)[-1]
        plt.plot(list(np.arange(len(dfg["Infected_count"])))+[MAX_DAY],list(100*dfg["Infected_count"].values/POPULATION)+[last_val],label=scenario_labels[i])
#plt.plot([0,70],[5,5],"--",c='grey')
plt.legend()
plt.xlabel("Days since outbreak")
plt.ylabel("Infected (% of Population)")
plt.subplots_adjust(right=0.98,left=0.08)
plt.savefig("analyze_simulation_output/infected_count_comparison.png")


#### comparison of deaths
plt.figure(figsize=FIGSIZE)
for i in range(len(scenario_labels)):
    if True:#i in [1,3,4]:
        simulation_file = "simulation_output/"+scenario_filenames[i]+".csv"
        df = pd.read_csv(simulation_file)
        dfg = df.groupby("Date").mean()
        last_val = (100 * dfg["Death_count"].values / POPULATION)[-1]
        plt.plot(list(np.arange(len(dfg["Death_count"])))+[MAX_DAY],list(100*dfg["Death_count"].values/POPULATION)+[last_val],label=scenario_labels[i])
#plt.plot([0,70],[5,5],"--",c='grey')
plt.legend()
plt.xlabel("Days since outbreak")
plt.ylabel("Deceased (% of Population)")
plt.subplots_adjust(right=0.98,left=0.08)
plt.savefig("analyze_simulation_output/death_count_comparison.png")

#### comparison of recoveries
plt.figure(figsize=FIGSIZE)
for i in range(len(scenario_labels)):
    if True:#i in [1,3,4]:
        simulation_file = "simulation_output/"+scenario_filenames[i]+".csv"
        df = pd.read_csv(simulation_file)
        dfg = df.groupby("Date").mean()
        last_val = (100 * dfg["Recovered_count"].values / POPULATION)[-1]
        plt.plot(list(np.arange(len(dfg["Recovered_count"])))+[MAX_DAY],list(100*dfg["Recovered_count"].values/POPULATION)+[last_val],label=scenario_labels[i])
#plt.plot([0,70],[5,5],"--",c='grey')
plt.legend()
plt.xlabel("Days since outbreak")
plt.ylabel("Recovered (% of Population)")
plt.subplots_adjust(right=0.98,left=0.08)
plt.savefig("analyze_simulation_output/recovered_count_comparison.png")

#### comparison of number of notifications
try:
    plt.figure(figsize=FIGSIZE)
    for i in range(len(scenario_labels)):
        if True:#i in [1,3,4]:
            simulation_file = "simulation_output/"+scenario_filenames[i]+".csv"
            df = pd.read_csv(simulation_file)
            dfg = df.groupby("Date").mean()
            last_val = (100*dfg["notified_count"].values/POPULATION)[-1]
            plt.plot(list(np.arange(len(dfg["notified_count"])))+[MAX_DAY],list(100*dfg["notified_count"].values/POPULATION)+[last_val],label=scenario_labels[i])
    #plt.plot([0,70],[5,5],"--",c='grey')
    plt.legend()
    plt.xlabel("Days since outbreak")
    plt.ylabel("% of population notified to isolate")
    plt.subplots_adjust(right=0.98, left=0.08)
    plt.savefig("analyze_simulation_output/notified_count_comparison.png")
except Exception as e:
    pass

# compare locked zones
try:
    plt.figure(figsize=FIGSIZE)
    for i in range(len(scenario_labels)):
        if True:#i in [1,3,4]:
            simulation_file = "simulation_output/"+scenario_filenames[i]+".csv"
            df = pd.read_csv(simulation_file)
            dfg = df.groupby("Date").mean()
            last_val = (dfg["locked_zones"].values)[-1]
            plt.plot(list(np.arange(len(dfg["locked_zones"])))+[MAX_DAY],list(dfg["locked_zones"].values)+[last_val],label=scenario_labels[i])
    plt.legend()
    plt.xlabel("Days since outbreak")
    plt.ylabel("Zone ID")
    plt.subplots_adjust(right=0.98, left=0.08)
    plt.savefig("analyze_simulation_output/locked_zones_comparison.png")
except Exception as e:
    pass

# number of entities per zone:
try:
    simulation_file = "simulation_output/" + scenario_filenames[0]
    df = pd.read_csv(simulation_file+"_overall_agent_status.csv")
    df["Date_Time"]=pd.to_datetime(df["Date_Time"])
    dfg = df.query("Date_Time >'2020-01-01' and Date_Time <'2020-01-03'").drop_duplicates("currentLocationID")
    residential_counts_zone = dfg.query("currentLocationType == 'residential'").groupby("zone_id").count()["id"].values
    employment_counts_zone = dfg.query("currentLocationType == 'employment'").groupby("zone_id").count()["id"].values
    school_counts_zone = dfg.query("currentLocationType == 'school'").groupby("zone_id").count()["id"].values
    shopping_mall = dfg.query("currentLocationType == 'shopping_mall'").groupby("zone_id").count()["id"].values
    zone_counts = np.vstack((residential_counts_zone,
               employment_counts_zone,
               shopping_mall,
               school_counts_zone)).T
    zone_counts = pd.DataFrame(zone_counts,columns=["residential","employment","shopping_mall","school"])
    zone_counts.to_csv("analyze_simulation_output/locations_per_zone.csv")
except Exception as e:
    print("error:",e)


# analyse individual user dataset

SHOPPING_MALL_VISITS = []
plt.figure(figsize=FIGSIZE)
for i in range(len(scenario_labels)):
    simulation_file = "simulation_output/" + scenario_filenames[i]
    df1=pd.read_csv(simulation_file+"_overall_agent_status.csv")
    df1["Date_Time"]= pd.to_datetime(df1["Date_Time"])
    df1["date"]=pd.to_datetime(df1.loc[:,"Date_Time"]).dt.date

    #df1_gp = df1.groupby(["date", "currentLocationType"]).count()
    df1_gp = df1.groupby(["date", "currentLocationType"])["id"].nunique().reset_index() # unique subs per entity per day
    #df1_gp = df1_gp.reset_index()
    df1_gp["date"] = pd.to_datetime(df1_gp["date"])

    df1_gp_residential = df1_gp.query("currentLocationType == 'residential'").loc[:, ["date", "currentLocationType"]]
    df1_gp_residential = df1_gp_residential.rename(columns={"currentLocationID": "residential"})
    df_shopping_mall_market_visits=df1_gp.query("currentLocationType == 'shopping_mall' or currentLocationType == 'market'").groupby(
        "date").mean().reset_index().loc[:, ["date", "id"]]
    #df1_gp_shopping_mall = df1_gp.query("currentLocationType == 'shopping_mall'").loc[:, ["date", "currentLocationID"]]
    #df1_gp_shopping_mall = df1_gp_shopping_mall.rename(columns={"currentLocationID": "shopping_mall"})
    df_shopping_mall_market_visits = df_shopping_mall_market_visits.rename(columns={"id": "shopping_mall_market_visits"})
    df1_gp_entity_visits = df1_gp_residential.merge(df_shopping_mall_market_visits, how="left", on="date")
    df1_gp_entity_visits = df1_gp_entity_visits.fillna(0)

    avg_daily_number_of_shopping_mall_visitors = int(np.average((df1_gp_entity_visits["shopping_mall_market_visits"]).values))
    plt.plot(df1_gp_entity_visits["shopping_mall_market_visits"].values[:-1],label=scenario_labels[i])
    print(avg_daily_number_of_shopping_mall_visitors)
    SHOPPING_MALL_VISITS.append(avg_daily_number_of_shopping_mall_visitors)

plt.legend()
plt.xlabel("Days since outbreak")
plt.ylabel("Shopping mall and market visitors")
plt.subplots_adjust(right=0.98,left=0.08)
plt.ylim([0,4000])
plt.savefig("analyze_simulation_output/shopping_mall_visits.png")

POPULATIONS = []
INFECTIONS = []
DEATHS = []
RECOVERIES = []
for i in range(len(scenario_labels)):
    simulation_file = "simulation_output/" + scenario_filenames[i] + ".txt"
    with open(simulation_file) as txt_file:
        str_txt = txt_file.readlines()

    for line in str_txt:
        if line.find("Number of Infected = ") != -1 :
            INFECTIONS.append(line.split("Number of Infected = ")[1].split(" ")[0])
        if line.find("Number of Recovered = ") != -1 :
            RECOVERIES.append(line.split("Number of Recovered = ")[1].split(" ")[0])
        if line.find("Number of Deaths = ") != -1 :
            DEATHS.append(line.split("Number of Deaths = ")[1].split(" ")[0])
        if line.find("Total Population = ") != -1:
            POPULATIONS.append(line.split("Total Population = ")[1].split()[0])

df = pd.DataFrame(np.vstack((scenario_labels,POPULATIONS,INFECTIONS,RECOVERIES,DEATHS,SHOPPING_MALL_VISITS)).T,columns=["Scenario","Population","Number_infected","Number_recovered","Number_deceased","shopping_mall_visits"])
df.to_csv("analyze_simulation_output/overall_comparison.csv")


