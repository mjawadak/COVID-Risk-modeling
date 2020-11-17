print("loading packages")
from scipy import integrate
from getRiskIndividual import get_geo_risk, get_risk_generic
from mobility_model import getNextLocType
from RecoveryModel import RecoveryModel
import numpy as np
import pandas as pd
import datetime
import time
import sys
import json
from sklearn.cluster import KMeans
from scipy.stats import lognorm
from getRiskIndividual import f_age


if len(sys.argv) == 1:
    config_file_loc = 'config.json'
else:
    config_file_loc = sys.argv[1]
print(config_file_loc)


def get_scenario_text(config):
    text_arr = []
    if config["INTERVENTION_INFO"]["flag"] == True:
        text_arr.append("Lockdown enabled")
    if config["SELF_ISOLATION"]["flag"] == True and config["HOSPITALIZATION"]["flag"] == True:
        text_arr.append("Self Isolation")
    elif config["SELF_ISOLATION"]["flag"] == True:
        text_arr.append("Self Isolation (without Hosp.)")
    else:
        text_arr.append("Hosp. enabled")
    if config["MASK_COMPLIANCE"]["flag"] == True:
        text_arr.append("Mask Compliance ({})".format(config["MASK_COMPLIANCE"]["COMPLIANCE_RATE"]))
    if config["CONTACT_TRACING"]["flag"] == True:
        text_arr.append("Contact Tracing ({})".format(config["CONTACT_TRACING"]["COMPLIANCE_RATE"]))
    if len(text_arr) == 0:
        text_arr.append("No Intervention & No Precaution")

    return ",".join(text_arr)


np.random.seed(100)

# command to run in bash shell: nohup python3.6 run_simulation.py > run.log 1> run1.log 2> run2.log &

TS = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
print("loaded the packages")

age_risk_list = []

for a in np.arange(90):
    if a <= 80:
        age_risk_list.append(np.max([f_age(a), 0]))
    else:
        age_risk_list.append(1.0)

# morbidity_infection_risk_factors = [1.0,1.3,1.3,1.4,1.6,1.7,1.7,2.0]

POP_DIST = [0.25 * 9 / 14, 0.17, 0.42, 0.09, 0.07]  # 0-14,15-24,25-54,55-64,65-90
POP_DIST = POP_DIST / np.sum(POP_DIST)


def get_age():
    # population distribution of indonesia
    POP_BIN_MIN = [5, 15, 25, 55, 65]
    POP_BIN_MAX = [14, 24, 54, 64, 80]
    hist_bin = np.random.choice(np.arange(len(POP_DIST)), p=POP_DIST)
    return np.random.randint(POP_BIN_MIN[hist_bin], POP_BIN_MAX[hist_bin])







config_str = ""
with open(config_file_loc, "r") as config_file:
    config_str = config_file.read()
with open(config_file_loc, "r") as config_file:
    config = json.load(config_file)["parameters"]

SCALE_FACTOR = config["SCALE_FACTOR"]
NUM_RESIDENCES = int(config["entity_counts"]["residential"] / float(SCALE_FACTOR))
NUM_SCHOOLS = int(config["entity_counts"]["school"] / float(SCALE_FACTOR))
NUM_RETAILS = int(config["entity_counts"]["retail"] / float(SCALE_FACTOR))
NUM_WORKPLACES = int(config["entity_counts"]["employment"] / float(SCALE_FACTOR))
NUM_HOSPITALS = int(config["entity_counts"]["hospital"] / float(SCALE_FACTOR))
NUM_MARKETS = int(config["entity_counts"]["market"] / float(SCALE_FACTOR))
POPULATION = int(config["POPULATION"] / float(SCALE_FACTOR))
INTERVENTION_INFO = config["INTERVENTION_INFO"]
_d = np.array(list(INTERVENTION_INFO["locked_entities"].keys()))
LOCKED_ENTITIES = _d[list(INTERVENTION_INFO["locked_entities"].values())].tolist()
SAVE_AGENT_LIST = config["SAVE_AGENT_LIST"]["flag"]
HOURS_TO_SAVE = config["SAVE_AGENT_LIST"]["hours_to_save"]
COMPLIANCE_RATE = float(config["COMPLIANCE_RATE"])
ZONE_INTERVENTION_INFO = config["ZONE_INTERVENTION_INFO"]
ZONE_THRESHOLD_INFECTED = float(ZONE_INTERVENTION_INFO["ZONE_THRESHOLD_INFECTED"])
ZONE_LOCK_DUR_DAYS = float(ZONE_INTERVENTION_INFO["ZONE_LOCK_DUR_DAYS"])
NUM_ZONES = int(config["NUM_ZONES"])
SELF_ISOLATION = config["SELF_ISOLATION"]
CONTACT_TRACING = config["CONTACT_TRACING"]
HOURS = config["NUMBER_ITERATIONS_PER_DAY"]
MASK_INFECTION_FACTOR = config["MASK_COMPLIANCE"]["MASK_INFECTION_FACTOR"]  # [1, 0.35]
NUMBER_ITERATIONS_PER_DAY = len(HOURS)
if NUMBER_ITERATIONS_PER_DAY == 0:
    NUMBER_ITERATIONS_PER_DAY = 24
    HOURS_TO_ADD = np.ones(24)
else:
    HOURS_TO_ADD = list(np.diff(HOURS)) + [24 - HOURS[-1]]
MAX_DAYS_OF_INFECTION = config["MAX_DAYS_OF_INFECTION"]  # 30#12#30
INFECTION_RATE_FACTOR = config["INFECTION_RATE_FACTOR"]  # 5#1#5

ENTITY_NAMES = list(config["entity_counts"].keys())
ENTITY_NUM = list(config["entity_counts"].values())
ENTITY_NUM_CUM = [0] + list(np.cumsum(ENTITY_NUM))
ENTITY_TYPE_TO_ID = {}
for i in range(len(ENTITY_NUM_CUM) - 1):
    ENTITY_TYPE_TO_ID[ENTITY_NAMES[i]] = [ENTITY_NUM_CUM[i], ENTITY_NUM_CUM[i + 1]]

TOTAL_ENTITIES = np.sum(ENTITY_NUM)

# incubation period parameters
meanlog = config["meanlog"]  # 1.798#1.644 # 95% CI: 1.495–1.798dsi
sdlog = config["sdlog"]  # 0.521#0.363 # 95% CI: 0.201–0.521
incubation_dist = lognorm(s=[sdlog], scale=np.exp(meanlog))


### Comorbidity parameters
'''
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7314621/
Multiple 1.0
Hypertension 0.49
Obesity 0.48
Respiratory 0.34
Diabetes 0.28
Cardiovascular 0.28 
None 0
morbidity_infection_risk_factors = 1.0,1.28,1.28,1.34,1.48,1.49,2.0 # None,Cardiovascular,Diabetes,Respiratory,Obesity,Hypertension,Multiple
'''
population_with_no_condition_table = config["MORBIDITY"]["population_with_no_condition_table"]
age_grp_no_condition = np.array([i for i in map(lambda x: np.array(x.split("-")).astype(int), population_with_no_condition_table.keys())])
age_grp_no_condition = np.concatenate((age_grp_no_condition, np.array([list(population_with_no_condition_table.values())]).T), axis=1)
morbidity_infection_risk_factors = config["MORBIDITY"]["morbidity_infection_risk_factors"]
morbidity_death_risk_factors = morbidity_infection_risk_factors

TYPE_OF_JOB = config["TYPE_OF_JOB"]["type"]
INCOME_MEANS = np.arange(15000, -1500, -1500)
INCOME_MEANS[-1] = 500

##################################################### Infectious Model #####################################################
w_shape = config["w_shape"]  # 1.75#2.83#1.75#2.83 # 1.75-4.7
w_scale = config["w_scale"]  # 6.9#5.67#6.9#5.67 # 4.7- 6.9

scenario_text = get_scenario_text(config)

SIM_PARAMS = str(int(config["MASK_COMPLIANCE"]["flag"])) + str(int(INTERVENTION_INFO["flag"])) + str(
    int(ZONE_INTERVENTION_INFO["flag"])) + str(int(CONTACT_TRACING["flag"])) + str(int(SELF_ISOLATION["flag"])) + str(int(config["HOSPITALIZATION"]["flag"]))

print(scenario_text, "simulation_" + SIM_PARAMS + "_" + TS)


def w(x):
    return (w_shape / w_scale) * (x / w_scale) ** (w_shape - 1) * np.exp(-(x / w_scale) ** w_shape)


def getInfectionProb_daily(days_since_infection):
    if days_since_infection == 0:
        return 0
    return integrate.quad(w, days_since_infection - 1, days_since_infection)[0]


def getInfectionProb_hourly(days_since_infection_hourly_fraction):
    if days_since_infection_hourly_fraction == 0:
        return 0
    return integrate.quad(w, days_since_infection_hourly_fraction - (1 / 24.0), days_since_infection_hourly_fraction)[0]


I = []
for i in np.arange(0., MAX_DAYS_OF_INFECTION + 3, 1 / 24.):
    I.append(getInfectionProb_hourly(i))
I = np.array(I)


def getInfectionProb_hourly_cached_np(days_since_infection_hourly_fraction):
    d = days_since_infection_hourly_fraction
    return I[(np.round(d) * 24 + (d - np.round(d)) * 24).astype(int)] / INFECTION_RATE_FACTOR


##########################################################################################################


LOCKED_ENTITY_IDS = []

notified_count = 0


def get_infected_count():
    counter = 0
    for a in agent_list:
        if a.is_infected == True:
            counter += 1
    return counter


def get_infected_count_per_entity(entity):
    infected_count_per_entity = 0
    for ei in entity.current_list_of_agent_ids:
        if agent_list[ei].is_infected == True:
            infected_count_per_entity += 1
    return infected_count_per_entity


def get_death_count():
    counter = 0
    for a in agent_list:
        if a.has_died == True:
            counter += 1
    return counter


def get_agent_list_all(ts):
    agent_list_current_ts = []

    for i in range(len(agent_list)):
        agent_list_current_ts.append(agent_list[i].get_agent_variables(ts))
    return agent_list_current_ts


overall_agent_status_columns = ["Date_Time",
                                    "id",
                                    "age",
                                    "gender",
                                    "infection_status",
                                    "days_since_infection",
                                    "date_of_infection",
                                    "date_of_symptoms",
                                    "date_of_recovery",
                                    "date_of_death",
                                    "date_hospital_check_in",
                                    "date_hospital_check_out",
                                    "residence_id",
                                    "school_id",
                                    "workplace_id",
                                    "currentLocationID",
                                    "currentLocationType",
                                    "lat",
                                    "lon",
                                    "zone_id",
                                    "is_notified_to_isolate",
                                    "is_symptomatic",
                                    "incubation_period",
                                    "days_since_isolation",
                                    "isolation_times",
                                    "days_since_symptomatic",
                                    "in_hospital",
                                    "wears_mask",
                                    "type_of_job",
                                    "morbidity",
                                    "exposure_time",
                                    "infected_from_user_id",
                                    "transmitted_count",
                                    "asymptomatic_transmission",
                                    "locationID_of_infection",
                                    "income",
                                    "household_size",
                                    "bmi",
                                    "date_of_test",
                                    "test_result"
                                    ]


class Agent():
    def __init__(self, id, person_id=None, age=None, gender=None):
        self.id = id
        if person_id == None:
            self.person_id = self.id
        else:
            self.person_id = person_id
        if age == None:
            self.age = get_age()  # np.random.randint(5, 90)
        else:
            self.age = age
        if gender == None:
            self.gender = np.random.randint(0, 2)  # np.random.choice(["female", "male"]) # 0->female,1->male
        else:
            # print(gender,len(gender))
            self.gender = 0 if "Female" in gender else 1
        self.is_infected = False
        self.has_died = False
        self.has_recovered = False
        self.days_since_infection = 0
        self.date_of_infection = ""
        self.date_of_symptoms = ""
        self.date_of_recovery = ""
        self.date_of_death = ""
        self.date_hospital_check_in = ""
        self.date_hospital_check_out = ""
        self.residence_id = get_random_entity_index("residential")
        self.school_id = get_random_entity_index("school")
        self.workplace_id = get_random_entity_index("employment")
        self.currentLocationID = self.residence_id
        self.currentLocationType = "residential"
        self.lat = entity_list[self.currentLocationID].lat
        self.lon = entity_list[self.currentLocationID].lon
        self.zone_id = entity_list[self.currentLocationID].zone_id
        self.is_notified_to_isolate = False
        self.is_symptomatic = 0
        self.incubation_period = incubation_dist.rvs()
        self.past_contacts = []
        self.days_since_isolation = 0
        self.isolation_times = 0
        self.days_since_symptomatic = 0
        self.in_hospital = 0
        self.wears_mask = 0
        if self.age >= 22 and self.age < 60:
            self.type_of_job = np.random.choice(["1", "2", "3"],
                                                p=config["TYPE_OF_JOB"]["prob"])  # ["Other job", "Remote worker","Health worker"]
            if TYPE_OF_JOB[self.type_of_job] == "Health worker":
                self.workplace_id = get_random_entity_index("hospital")
            if TYPE_OF_JOB[self.type_of_job] == "Remote worker":
                self.workplace_id = self.residence_id
        else:
            self.type_of_job = "0"  # "No job"


        if config["ASYMPTOMATIC_RATIO_IN_RECOVERY_MODEL"]["flag"] == False:
            self.max_days_of_infection = np.random.randint(12, MAX_DAYS_OF_INFECTION + 1)
        else:
            if np.random.random() < config["ASYMPTOMATIC_RATIO_IN_RECOVERY_MODEL"]["ratio"]:
                self.max_days_of_infection = np.random.uniform(0, self.incubation_period)
            else:
                if self.incubation_period < MAX_DAYS_OF_INFECTION:
                    self.max_days_of_infection = np.random.uniform(self.incubation_period, MAX_DAYS_OF_INFECTION)
                else:
                    self.max_days_of_infection = MAX_DAYS_OF_INFECTION


        pop_no_condition = age_grp_no_condition[(self.age >= age_grp_no_condition[:, 0]) & (self.age < age_grp_no_condition[:, 1])][0, 2]

        pop_dist_morbidity = [pop_no_condition] + list(
            np.ones(len(morbidity_infection_risk_factors) - 1) * (1 - pop_no_condition) / (
                        len(morbidity_infection_risk_factors) - 1))

        self.morbidity = np.random.choice(
            list(morbidity_infection_risk_factors.keys()),
            p=pop_dist_morbidity)

        self.morbidity_infection_risk_factor = morbidity_infection_risk_factors[
            self.morbidity]  # [1.0,1.3,1.3,1.4,1.6,1.7,1.7,2.0]
        self.exposure_time = 0
        self.infected_from_user_id = -1
        self.transmitted_count = 0
        self.asymptomatic_transmission = -1
        self.locationID_of_infection = -1
        self.income = 0
        self.household_size = 0
        self.bmi = 0
        if self.morbidity == "Obesity":
            self.bmi = np.random.randint(30, 42)
        else:
            self.bmi = np.random.randint(19, 32)
        self.date_of_test = ""
        self.test_result = -1

    def update_infection_state(self):
        global notified_count, CONTACT_TRACING, current_date
        if self.is_infected == True:
            if self.days_since_infection > self.max_days_of_infection:
                self.date_of_recovery = current_date
                self.has_recovered = True
                self.is_notified_to_isolate = False
                self.is_infected = False
                self.is_symptomatic = 0
                if self.in_hospital == 1:
                    self.date_hospital_check_out = current_date
                self.in_hospital = 0
                self.days_since_infection = 0
                infected_list.remove(self.id)
                recovered_list.append(self.id)
            else:
                self.days_since_infection += 1.0 / 24.0
                if self.is_symptomatic == 1:
                    self.days_since_symptomatic += 1.0 / 24.0
                if self.days_since_infection > self.incubation_period and self.is_symptomatic == 0:
                    self.is_symptomatic = 1
                    self.date_of_symptoms = current_date

                    if CONTACT_TRACING["flag"] == True:  # notify all past contacts to isolate
                        for p in self.past_contacts:
                            for a_index in p:
                                if agent_list[a_index].is_notified_to_isolate == False and agent_list[
                                    a_index].isolation_times == 0:
                                    notified_count = notified_count + 1
                                    agent_list[a_index].isolation_times += 1
                                agent_list[a_index].is_notified_to_isolate = True


        elif CONTACT_TRACING["flag"] == True and self.is_notified_to_isolate == True:
            self.days_since_isolation += 1.0 / 24.0
            # print("self.days_since_isolation",self.days_since_isolation)
            if self.days_since_isolation > CONTACT_TRACING["ISOLATION_PERIOD"]:
                self.is_notified_to_isolate = False

    def get_agent_variables(self, ts):
        infection_status = "susceptible"
        if self.is_infected == True:
            infection_status = "infected"
        if self.has_died == True:
            infection_status = "deceased"
        if self.has_recovered == True:
            infection_status = "recovered"

        currentLocationType = self.currentLocationType
        if currentLocationType == "retail":
            currentLocationType = "shopping_mall"

        if config["get_risk"] == True:
            proximity = len(entity_list[self.currentLocationID].current_list_of_agent_ids)
            proximity_infected = entity_list[self.currentLocationID].count_of_infected
            risk, age_factor, gender_factor, proximity_factor, infected_individuals_factor, infectiousness_factor = get_geo_risk(
                self.age, self.gender, self.days_since_infection, proximity, proximity_infected)
        else:
            risk = -1

        return [ts.strftime("%Y-%m-%d %H:%M:%S"),
                self.person_id,  # self.id,
                self.age,
                self.gender,
                infection_status,
                self.days_since_infection,
                str(self.date_of_infection),
                str(self.date_of_symptoms),
                str(self.date_of_recovery),
                str(self.date_of_death),
                str(self.date_hospital_check_in),
                str(self.date_hospital_check_out),
                self.residence_id,
                self.school_id,
                self.workplace_id,
                self.currentLocationID,
                currentLocationType,
                self.lat,
                self.lon,
                self.zone_id,
                int(self.is_notified_to_isolate),
                int(self.is_symptomatic),
                self.incubation_period,
                self.days_since_isolation,
                self.isolation_times,
                self.days_since_symptomatic,
                self.in_hospital,
                self.wears_mask,
                TYPE_OF_JOB[self.type_of_job],
                self.morbidity,
                self.exposure_time,
                self.infected_from_user_id,
                self.transmitted_count,
                self.asymptomatic_transmission,
                self.locationID_of_infection,
                self.income,
                self.household_size,
                self.bmi,
                str(self.date_of_test),
                self.test_result
                ]


class Entity():
    def __init__(self, cat, lat, lon, zone_id):
        self.category = cat
        self.lat = lat
        self.lon = lon
        self.zone_id = zone_id
        self.current_list_of_agent_ids = []
        self.count_of_infected = 0


def get_random_entity_index(loc_type):
    return np.random.randint(ENTITY_TYPE_TO_ID[loc_type][0],
                             ENTITY_TYPE_TO_ID[loc_type][1])  # sample from min and max values


def perform_testing(current_datehour_in_simulation):
    agent_ids_selected_for_testing = np.random.randint(0, POPULATION,
                                                       int(POPULATION * config["TESTING_INFO"]["daily_rate"]))
    for id in agent_ids_selected_for_testing:
        agent_list[id].date_of_test = current_datehour_in_simulation
        if agent_list[id].is_infected == True:
            agent_list[id].test_result = 1
            if agent_list[id].is_symptomatic == 0:
                agent_list[id].is_symptomatic = 1
                agent_list[id].date_of_symptoms = current_datehour_in_simulation
        else:
            agent_list[id].test_result = 0


def update_locations(current_datehour_in_simulation, COMPLIANCE_RATE, contact_tracing_enabled):
    # for e in entity_list:
    # e.current_list_of_agent_ids = []
    t_next_loc_type = 0
    t21 = 0
    t32 = 0
    t43 = 0
    t54 = 0
    t3a_3 = 0
    t4_3a = 0
    for_compliance = np.random.uniform(0.0, 1.0, POPULATION)
    for_self_isolation = np.random.uniform(0.0, 1.0, POPULATION)
    for_contact_tracing = np.random.uniform(0.0, 1.0, POPULATION)
    for_hospitalization = np.random.uniform(0.0, 1.0, POPULATION)
    for_getNextLocType = np.random.uniform(0.0, 1.0, POPULATION)
    for_lockdown_intensity = np.random.uniform(0.0, 1.0, POPULATION)

    for i in range(len(agent_list)):
        t1 = time.time()

        next_loc_type = getNextLocType(agent_list[i].age, current_datehour_in_simulation, for_getNextLocType[i])
        t2 = time.time()
        t_next_loc_type += t2 - t1

        if (next_loc_type != agent_list[i].currentLocationType or agent_list[i].currentLocationID == -1) and agent_list[
            i].has_died == False:

            #### for contact tracing
            if CONTACT_TRACING["flag"] == True and infected_list.__len__() > CONTACT_TRACING["threshold_infected"] and (
                    current_datehour_in_simulation.hour == 0 or current_datehour_in_simulation.hour == 8 or current_datehour_in_simulation.hour == 16 or current_datehour_in_simulation.hour == 19):
                agent_list[i].past_contacts.append(
                    entity_list[agent_list[i].currentLocationID].current_list_of_agent_ids)
                if len(agent_list[i].past_contacts) > CONTACT_TRACING["PAST_DAYS"] * 5:
                    agent_list[i].past_contacts.pop(0)

            if agent_list[i].currentLocationID != -1:
                entity_list[agent_list[i].currentLocationID].current_list_of_agent_ids.remove(i)

            t3 = time.time()

            if next_loc_type == "residential":
                next_loc_id = agent_list[i].residence_id
            elif next_loc_type == "school":
                next_loc_id = agent_list[i].school_id
            elif next_loc_type == "employment":
                next_loc_id = agent_list[i].workplace_id
                if agent_list[i].type_of_job == 3:
                    next_loc_type = 'hospital'
                if agent_list[i].type_of_job == 2:
                    next_loc_type = 'residential'
            else:  # ony in case of retail, market
                next_loc_id = get_random_entity_index(next_loc_type)
            t3a = time.time()


            # for_compliance = np.random.uniform(0.0,1.0)
            if (
                    (intervention_in_effect_flag == True and next_loc_type in LOCKED_ENTITIES and
                     for_lockdown_intensity[i] < config["INTERVENTION_INFO"][
                         "COMPLIANCE_RATE"]) or  # for complete lockdown
                    next_loc_id in LOCKED_ENTITY_IDS or  # for zone/entity level lockdown
                    agent_list[i].residence_id in LOCKED_ENTITY_IDS or  # for zone level lockdown
                    (SELF_ISOLATION["flag"] == True and agent_list[i].is_symptomatic == True and for_self_isolation[i] <
                     SELF_ISOLATION["COMPLIANCE_RATE"]) or  # for self isolation
                    (CONTACT_TRACING["flag"] == True and contact_tracing_enabled == 1 and agent_list[
                        i].is_notified_to_isolate == True and for_contact_tracing[i] < CONTACT_TRACING[
                         "COMPLIANCE_RATE"]) or
                    (config["HOSPITALIZATION"]["flag"] == True and agent_list[i].is_symptomatic == 1)
            ) and for_compliance[i] <= COMPLIANCE_RATE:

                # for hospitals: probablistically, symptomatic people go to hospital based on age factor in risk profiling:
                if config["HOSPITALIZATION"]["flag"] == True and agent_list[i].is_symptomatic == 1 and (
                        agent_list[i].in_hospital == 1 or for_hospitalization[i] < age_risk_list[agent_list[i].age]):

                    if agent_list[i].in_hospital == 0:
                        agent_list[i].in_hospital = 1
                        agent_list[i].currentLocationID = get_random_entity_index("hospital")
                        agent_list[i].currentLocationType = "hospital"
                        agent_list[i].date_hospital_check_in = current_date

                else:
                    agent_list[i].currentLocationID = agent_list[i].residence_id
                    agent_list[i].currentLocationType = "residential"
            else:
                agent_list[i].currentLocationID = next_loc_id
                agent_list[i].currentLocationType = next_loc_type

            agent_list[i].lat = entity_list[agent_list[i].currentLocationID].lat
            agent_list[i].lon = entity_list[agent_list[i].currentLocationID].lon
            agent_list[i].zone_id = entity_list[agent_list[i].currentLocationID].zone_id

            entity_list[agent_list[i].currentLocationID].current_list_of_agent_ids.append(i)


def update_agents_after_contact(agents_in_entity_list):
    if len(agents_in_entity_list) > 0:
        t1 = time.time()
        infected_individuals = []
        df_infected = []  # a data frame that has age and gender of every individual
        for a in agents_in_entity_list:
            if agent_list[a].is_infected == True:
                infected_individuals.append(a)
                df_infected.append([agent_list[a].age, agent_list[a].gender])
        df_infected = np.array(df_infected)
        # df_infected = pd.DataFrame(df_infected, columns=["age", "gender"])
        # print(df_infected.shape)

        # Below code is for changing the state from susceptile to infected.
        # For change of state of infected to recovery, rely on the days since infection for every infected individual

        t2 = time.time()
        infection_prob = 0
        agent_id_with_max_infectiousness = -1
        _maxVal = 0
        for i in range(len(infected_individuals)):
            # infection_prob = getInfectionProb_daily(infected_individuals[i].days_since_infection)/24.0
            individual_prob = getInfectionProb_hourly_cached_np(
                agent_list[infected_individuals[i]].days_since_infection)
            if individual_prob >= _maxVal:
                _maxVal = individual_prob
                agent_id_with_max_infectiousness = infected_individuals[i]
            infection_prob += individual_prob

        # print("infection_prob",infection_prob)
        n_rand = np.random.random(len(agents_in_entity_list))
        for_infected_from = np.random.random(len(agents_in_entity_list))
        exposure_times = np.random.uniform(0, 3, len(agents_in_entity_list))
        exposure_time_risk_factors = get_risk_generic(5.29,
                                                      exposure_times)  # 5.29 parameter to get 0.99 risk at 1 hour.
        for j in range(len(agents_in_entity_list)):
            agent_list[agents_in_entity_list[j]].exposure_time = exposure_times[j]
            if agent_list[agents_in_entity_list[j]].is_infected == False and \
                    agent_list[agents_in_entity_list[j]].has_recovered == False and \
                    agent_list[agents_in_entity_list[j]].has_died == False:
                # n_rand = np.random.random()
                if n_rand[j] < MASK_INFECTION_FACTOR[agent_list[agents_in_entity_list[j]].wears_mask] * infection_prob * \
                        agent_list[agents_in_entity_list[j]].morbidity_infection_risk_factor * \
                        exposure_time_risk_factors[j]:
                    agent_list[agents_in_entity_list[j]].is_infected = True
                    agent_list[agents_in_entity_list[j]].date_of_infection = current_date
                    # id_transmitting_individual = agents_in_entity_list[int(for_infected_from[j] * (len(agents_in_entity_list) - 1))]  # random index of infected individuals
                    # print(id_transmitting_individual)
                    agent_list[agents_in_entity_list[j]].locationID_of_infection = agent_list[
                        agents_in_entity_list[j]].currentLocationID
                    # print(agent_list[agents_in_entity_list[j]].locationID_of_infection)
                    agent_list[agents_in_entity_list[
                        j]].infected_from_user_id = agent_id_with_max_infectiousness  # id_transmitting_individual

                    if agent_list[agent_id_with_max_infectiousness].is_symptomatic == 0:
                        agent_list[agents_in_entity_list[j]].asymptomatic_transmission = 1

                    infected_list.append(agents_in_entity_list[j])

                    agent_list[agent_id_with_max_infectiousness].transmitted_count += 1

        if df_infected.shape[0] > 0:
            death_probabilities = recovery_model.predictDeathProbs(df_infected)

            # Below code is for changing the state from infected to death

            r = np.random.random(len(infected_individuals))
            for i in range(len(infected_individuals)):
                if r[i] < death_probabilities[i] * agent_list[infected_individuals[i]].morbidity_infection_risk_factor:
                    if agent_list[infected_individuals[i]].has_died == False:
                        agent_list[infected_individuals[i]].has_died = True
                        agent_list[infected_individuals[i]].days_since_infection = 0
                        agent_list[infected_individuals[i]].is_infected = False
                        agent_list[infected_individuals[i]].has_recovered = False
                        agent_list[infected_individuals[i]].date_of_death = current_date

                        death_list.append(infected_individuals[i])
                        infected_list.remove(infected_individuals[i])

def start_wearing_masks():
    for agent in agent_list:
        agent.wears_mask = np.random.choice([0, 1], p=[1 - config["MASK_COMPLIANCE"]["COMPLIANCE_RATE"],
                                                       config["MASK_COMPLIANCE"]["COMPLIANCE_RATE"]])


def getZoneRiskStats(agent_list):

    ZONE_STATS = np.zeros(NUM_ZONES)
    for a in agent_list:
        if a.is_infected == True:
            ZONE_STATS[a.zone_id] += 1
    return ZONE_STATS



df_loc_main = pd.read_csv("data/spatial_model_opencellid_locations.csv")
df_loc = df_loc_main.sample(TOTAL_ENTITIES, random_state=0).loc[:, ["lat", "lon"]].reset_index()

print("building zones")
clf = KMeans(n_clusters=NUM_ZONES, random_state=0)
clf.fit(df_loc.loc[:, ["lat", "lon"]].values)
zones = clf.predict(df_loc.loc[:, ["lat", "lon"]].values)
df_loc["zone"] = zones
df_loc = df_loc.reset_index()

# plt.scatter(df_loc["lat"],df_loc["lon"],c=df_loc["zone"])
# plt.scatter(df_loc_retails["lat"],df_loc_retails["lon"])#,c=df_loc["zone"])
# plt.scatter(df_loc_workplaces["lat"],df_loc_workplaces["lon"])#,c=df_loc["zone"])
# plt.scatter(df_loc_residential["lat"],df_loc_residential["lon"])#,c=df_loc["zone"])
# plt.scatter(df_loc_schools["lat"],df_loc_schools["lon"])#,c=df_loc["zone"])
# plt.show()

# sys.exit()
# print(df_loc_main.query("dist_center<000"))
# for e in range(len(df_loc)):
# print(e,df_loc.loc[e,"lat"])

# df_google = pd.read_csv("/Users/mk250133/OneDrive - Teradata/WORK/COVID/CovidRiskModelling/M and S/get_geo_locations_googleapi/geo_locations_googleapi_jakarta_corrected.csv")


# df_retail = df_google.query("query == 'shopping malls' and dist_center < 10000").sample(NUM_RETAILS).reset_index()
# print(df_retail)
# print(df_google.shape)
# plt.scatter(df_retail["lat"].values,df_retail["long"].values)
# plt.show()


intervention_flag = INTERVENTION_INFO["flag"]
intervention_in_effect_flag = False  # gets true if the count of infected cases exceeds the given threshold
contact_tracing_enabled = 0
mask_wearing_enabled = 0
helper_counter = 0  # to change the above flag only once
# create entity list
entity_list = []
entity_infected_count = []

ZONE_ENTITIES = []
for i in range(NUM_ZONES):
    ZONE_ENTITIES.append([])
ENTITY_TYPES = {}

##new code to build the entity list:
print("building the entity list")
e = 0
for e_name in config["entity_counts"]:
    entity_count = config["entity_counts"][e_name]
    # print(e_name, entity_count)
    entity_types = []
    for j in range(entity_count):
        entity_list.append(Entity(e_name, df_loc.loc[e, "lat"], df_loc.loc[e, "lon"], df_loc.loc[e, "zone"]))
        ZONE_ENTITIES[df_loc.loc[e, "zone"]].append(e)
        entity_types.append(e)
        e = e + 1
    ENTITY_TYPES[e_name] = entity_types

'''
for i in range(NUM_RESIDENCES):
    #print(df_loc.loc[e,"lat"],df_loc.loc[e,"lon"],df_loc.loc[e,"zone"])
    entity_list.append(Entity("residential",df_loc.loc[e,"lat"],df_loc.loc[e,"lon"],df_loc.loc[e,"zone"]))
    ZONE_ENTITIES[df_loc.loc[e,"zone"]].append(e)
    entity_types.append(e)
    e += 1
ENTITY_TYPES["residential"] = entity_types
entity_types = []
for i in range(NUM_SCHOOLS):
    entity_list.append(Entity("school",df_loc.loc[e,"lat"],df_loc.loc[e,"lon"],df_loc.loc[e,"zone"]))
    ZONE_ENTITIES[df_loc.loc[e, "zone"]].append(e)
    entity_types.append(e)
    e += 1
ENTITY_TYPES["school"] = entity_types
entity_types = []
for i in range(NUM_RETAILS):
    entity_list.append(Entity("retail",df_loc.loc[e,"lat"],df_loc.loc[e,"lon"],df_loc.loc[e,"zone"]))
    ZONE_ENTITIES[df_loc.loc[e, "zone"]].append(e)
    entity_types.append(e)
    e += 1
ENTITY_TYPES["retail"] = entity_types
entity_types = []
for i in range(NUM_WORKPLACES):
    entity_list.append(Entity("employment",df_loc.loc[e,"lat"],df_loc.loc[e,"lon"],df_loc.loc[e,"zone"]))
    ZONE_ENTITIES[df_loc.loc[e, "zone"]].append(e)
    entity_types.append(e)
    e += 1
ENTITY_TYPES["employment"] = entity_types
entity_types = []'''

# create agent list
agent_list = []
infected_list = []
recovered_list = []
death_list = []
house_sizes = {}

print("building the agent list: starting from beginning of epidemic")
for i in range(POPULATION):
    agent_list.append(Agent(i))
    
    if house_sizes.get(agent_list[i].residence_id) == None:
        house_sizes[agent_list[i].residence_id] = 1
    else:
        house_sizes[agent_list[i].residence_id] += 1
    if i == 100:
        agent_list[i].age = 20  # to ensure that the person does not die
        agent_list[i].incubation_period = 12  # to ensure that the person does not isolate
        agent_list[i].is_infected = True
        agent_list[i].days_since_infection = 0
        agent_list[i].date_of_infection = str(datetime.datetime.strptime(config["start_date"], '%Y-%m-%d'))
        infected_list.append(i)
    entity_list[agent_list[i].currentLocationID].current_list_of_agent_ids.append(i)

for i in range(POPULATION):
    # print(i)
    agent_list[i].household_size = house_sizes[agent_list[i].residence_id]
    # print(INCOME_MEANS)
    # print(agent_list[i].household_size)
    income_bin = agent_list[i].household_size
    if income_bin >= 10:
        income_bin = 10
    agent_list[i].income = np.random.normal(INCOME_MEANS[income_bin], INCOME_MEANS[income_bin] / 10)
    if agent_list[i].income < 0:
        agent_list[i].income = np.random.uniform(1, 50)

recovery_model = RecoveryModel(MAX_DAYS_OF_INFECTION,NUMBER_ITERATIONS_PER_DAY)  

ZONE_STATS_AVG = []
overall_agent_status = []
count_infected = get_infected_count()
count_death = get_death_count()
print("Starting simulations:", "Before infected_count = ", count_infected, ", death_count = ", count_death)
# current_date = datetime.datetime(year=2020,month=1,day=1,hour=0)
start_date = datetime.datetime.strptime(config["start_date"], '%Y-%m-%d')
current_date = datetime.datetime(year=start_date.year, month=start_date.month, day=start_date.day, hour=0)
total_time = 0
OUTPUT = []
cumulative_infected_count = 0
intervention_day = 0
lift_intervention_day = 0
LOCKED_ZONES = []
zone_lock_flag = 0
last_zone_lock_day = 0
highest_zone_risk = 0
enable_testing_flag = 0

for t in range(3600 * 24):  # test the code

    ### for zone level locking
    if ZONE_INTERVENTION_INFO["flag"] == True:
        if current_date.hour == 0:
            ZONE_STATS_AVG = []
        elif current_date.hour == 8:
            ZONE_STATS_AVG = getZoneRiskStats(agent_list)
        elif current_date.hour == 19:
            ZONE_STATS_AVG = ZONE_STATS_AVG + getZoneRiskStats(agent_list)
        elif current_date.hour == 22:
            ZONE_STATS_AVG = ZONE_STATS_AVG / 2.0
            zone_with_highest_risk = np.argmax(
                ZONE_STATS_AVG)  # only lock the zone with highest risk (according to infected)
            highest_zone_risk = ZONE_STATS_AVG[zone_with_highest_risk]

            if zone_lock_flag == 0:
                if ZONE_STATS_AVG[zone_with_highest_risk] > ZONE_THRESHOLD_INFECTED:
                    if ZONE_INTERVENTION_INFO["ZONE_ID"] == [] or ZONE_INTERVENTION_INFO["SEQUENTIAL"] == True:
                        locked_zone_id = [zone_with_highest_risk]
                    else:
                        locked_zone_id = ZONE_INTERVENTION_INFO["ZONE_ID"]

                    LOCKED_ZONES = locked_zone_id  # .append(locked_zone_id)
                    for l in locked_zone_id:
                        LOCKED_ENTITY_IDS = LOCKED_ENTITY_IDS + ZONE_ENTITIES[l]  # append the arrays
                    zone_lock_flag = 1
                    last_zone_lock_day = current_date
            elif (current_date - last_zone_lock_day).days > ZONE_LOCK_DUR_DAYS:
                LOCKED_ZONES = []
                LOCKED_ENTITY_IDS = []
                if ZONE_INTERVENTION_INFO["SEQUENTIAL"] == True:
                    zone_lock_flag = 0  # repeat the process, i.e. lock the zone with the highest number of cases

            print(ZONE_STATS_AVG, LOCKED_ZONES)
    ###

    if infected_list.__len__() > CONTACT_TRACING["threshold_infected"]:
        contact_tracing_enabled = 1

    if config["MASK_COMPLIANCE"]["flag"] == True:
        if infected_list.__len__() > config["MASK_COMPLIANCE"]["threshold"]:
            if mask_wearing_enabled == 0:
                start_wearing_masks()  # people start to wear masks
                mask_wearing_enabled = 1


    t1 = time.time()

    update_locations(current_date, COMPLIANCE_RATE, contact_tracing_enabled)
    t2 = time.time()
    ############################################ UPDATE LOCATIONS ###########################################
    entity_infected_count_list = []
    for e in entity_list:
        update_agents_after_contact(e.current_list_of_agent_ids)
        if config["get_risk"] == True:
            e.count_of_infected = get_infected_count_per_entity(e)
    #########################################################################################################

    t3 = time.time()
    ############################################ UPDATE INFECTION STATE #####################################
    for i in range(len(agent_list)):  # infected_list:
        agent_list[i].update_infection_state()
    #########################################################################################################

    ############################################ PERFORM TESTING ############################################
    if config["TESTING_INFO"]["flag"] == True:
        if infected_list.__len__() > config["TESTING_INFO"]["threshold_infected"]:
            enable_testing_flag = 1

        if enable_testing_flag == 1 and current_date.hour == 8:
            perform_testing(current_date)
    #########################################################################################################



    if infected_list.__len__() == 0:  # end simulation when all get recovered
        break

    if intervention_flag == True and helper_counter == 0 and infected_list.__len__() > INTERVENTION_INFO[
        "threshold_infected"]:
        intervention_day = current_date
        intervention_in_effect_flag = 1
        helper_counter == 1

    if intervention_flag == True and intervention_in_effect_flag == 1 and infected_list.__len__() < INTERVENTION_INFO[
        "lift_lockdown_threshold"]:
        lift_intervention_day = current_date
        _d = np.array(list(INTERVENTION_INFO["lockdown_entities_after_lift"].keys()))
        LOCKED_ENTITIES = _d[list(INTERVENTION_INFO["lockdown_entities_after_lift"].values())].tolist()
        intervention_in_effect_flag = 0

    t4 = time.time()

    total_time += t4 - t1

    OUTPUT.append([current_date.strftime("%Y-%m-%d"),
                   current_date.strftime("%H:%M:%S"),
                   infected_list.__len__(),
                   recovered_list.__len__(),
                   death_list.__len__(),
                   t4 - t1,
                   int(intervention_in_effect_flag),
                   "|".join(np.array(LOCKED_ZONES).astype(str)),
                   highest_zone_risk,
                   notified_count
                   ])



    if SAVE_AGENT_LIST == True and (current_date.hour in HOURS_TO_SAVE):
        overall_agent_status = overall_agent_status + get_agent_list_all(current_date)

    current_date = current_date + datetime.timedelta(hours=1)  # int(HOURS_TO_ADD[t%NUMBER_ITERATIONS_PER_DAY])

    # if current_date.hour ==0:
    print(current_date, ", infected =", infected_list.__len__(), ", recovered =", recovered_list.__len__(),
          ",deceased =",
          death_list.__len__())  # , ", update_locations = ",t2 - t1, ", update_status = ", t3 - t2,notified_count)




OUTPUT = pd.DataFrame(OUTPUT, columns=["Date", "Time",
                                       "Infected_count",
                                       "Recovered_count",
                                       "Death_count",
                                       "time_taken_secs",
                                       "intervention_in_effect_flag",
                                       "locked_zones",
                                       "highest_zone_risk",
                                       "notified_count"])
file_name = "OUTPUT_" + SIM_PARAMS + "_" + TS
OUTPUT.to_csv("simulation_output/" + file_name + ".csv")

if SAVE_AGENT_LIST == True:
    overall_agent_status = pd.DataFrame(overall_agent_status, columns=overall_agent_status_columns)
    overall_agent_status["Date_Time"] = pd.to_datetime(overall_agent_status["Date_Time"])
    overall_agent_status["Scenario"] = SIM_PARAMS + "_" + TS
    overall_agent_status.to_csv( "simulation_output/" + file_name + "_overall_agent_status.csv")


count_infected = get_infected_count()
count_death = get_death_count()
infected_count = infected_list.__len__() + recovered_list.__len__() + death_list.__len__()
recovered_count = recovered_list.__len__()
death_count = death_list.__len__()

unique_transmitted_individuals = 0
for agent in agent_list:
    if agent.transmitted_count > 0:
        unique_transmitted_individuals += 1
R0 = infected_count / float(unique_transmitted_individuals)

output_string = """
-------------------------------- REPORT ---------------------------------
Config = {}
Output file name = {}
Scenario = {}
Number of simulation hours run = {} ({} days)
Total run time in seconds = {} ({} average secs taken to simulate one interval)
Total Population = {}
Number of residences = {}
Number of schools = {}
Number of workplaces = {}
Number of retail shops = {}
SCALE_FACTOR = {}
Number of Infected = {} ({}% of total population)
Number of Recovered = {} ({}% of total population, {}% of total infected)
Number of Deaths = {} ({}% of total population, {}% of total infected)
R0 = {}
-------------------------------------------------------------------------
""".format(str(config),
           file_name,
           scenario_text,
           str(t),
           str(int(t / 24.0)),
           str(int(total_time)),
           "{:.2f}".format(total_time / t),
           str(POPULATION),
           str(NUM_RESIDENCES),
           str(NUM_SCHOOLS),
           str(NUM_WORKPLACES),
           str(NUM_RETAILS),
           str(SCALE_FACTOR),
           str(infected_count),
           str(round(100 * infected_count / float(POPULATION))),
           str(recovered_count),
           str(round(100 * recovered_count / float(POPULATION))),
           str(round(100 * recovered_count / float(infected_count))),
           str(death_count),
           str(round(100 * death_count / float(POPULATION))),
           str(round(100 * death_count / float(infected_count))),
           str(R0)
           )
with open("simulation_output/" + file_name + ".txt", "w") as outfile:
    outfile.write(output_string)


