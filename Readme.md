<header style="padding:10px;background:#f9f9f9;border-top:3px solid #00b2b1"><img id="Teradata-logo" src="https://www.teradata.com/Teradata/Images/Rebrand/Teradata_logo-two_color.png" alt="Teradata" width="220" align="right" />
 
# Covid360: Risk Modeling and Simulation Framework
</header>
We present the COVID360 solution that offers a set of risk models and a simulation tool that leverages several mathematical models to simulate the spread of COVID-19 in urban environments. Using COVID360, we can analyze the disease impact under different scenarios (e.g., lockdowns, self-isolation, and contact tracing).

This code allows us run the simulations and analyse the results.

To perform What-If analyses, execute step 1 with different configurations (save the scenario ids), and then use ```analyze_simulation_output.py``` to plot the results.

To calculate aggregated risks by location, execute Step 1 and then Step 3.

## Step 1: Run the simulations: 
To use the code we have first specify the parameters in the config.json file, which represents a given scenario, and then run the simulations by executing the run_simulation.py file locally on the machine.

```python run_simulation.py```

In the config.json file, we have several parameters. These parameters include the population size, the number of different locations, the epidemic model parameters and the parameters to specify different scenarios e.g., lockdowns, self-isolation and contact tracing. A description of these parameters is given below:

After the execution of the script, two files will be created in the simulation_output folder with the corresponding "Scenario_ID".

## Step 2: Analyse the results:

Change the "Scenario_ID" in ```analyze_simulation_output.py``` and execute it to generate the epidemiological curves for the number of active cases, recoveries, deaths and shopping mall visitors (retail and market) in the analyze_simulation_output folder.

```python analyze_simulation_output.py```

## Step 3 (optional): Compute and aggregate the individual risk scores by the locations in the simulation:

Change the "Scenario_ID" in ```analyze_simulation_output.py``` and execute it to generate the epidemiological curves for the number of active cases, recoveries, deaths and shopping mall visitors (retail and market) in the analyze_simulation_output folder.

```python consolidate_simulation_output_for_geo_risk.py```

This will generate a new file with the same name in simulation_ouput folder with a  suffix "overall_geo_status". This csv file corresponds to the aggregated risk scores that can be visualized on a map using any tool including Tableau or d3.js.

## Parameters in the config.json file: 
|Parameter Name|Parameter Description
|---|---|
|**start_date**| specify the start date of the simulation
|**entity_counts**| specifies the number for each of the 6 location types, which are residential, school, retail (shopping mall), employment, hospital and market.
|**POPULATION.count**| The number of individuals in the simulation.
|**POPULATION.distribution**| Population distribution. Make sure that the probabilities sum up to 1.
|**SCALE_FACTOR**| Not used. Used to scale down the population and the entity counts.
**INTERVENTION_INFO.flag**| A lockdown can be enforced by setting the **flag** parameter to true. 
**INTERVENTION_INFO.locked_entities**| The types of locations locked are specified in **locked_entities**; set the value to true to enable lockdown of that location type. 
**INTERVENTION_INFO.threshold_infected**| The lockdown comes into effect once the number of active cases exceeds **threshold_infected**. 
**INTERVENTION_INFO.lift_lockdown_threshold**| The lockdown is lifted after the active cases drop below **lift_lockdown_threshold**. 
**INTERVENTION_INFO.lockdown_entities_after_lift**| The entities kept in lockdown after the lift is lifted are in **lockdown_entities_after_lift**.
**INTERVENTION_INFO.COMPLIANCE_RATE**| The **COMPLIANCE_RATE** is a parameter that specifies the percentage of the population (to be set b/w 0 and 1) that complies with the lockdown instructions.
|**SAVE_AGENT_LIST.flag**| Set it to true if the output individual data needs to be saved in a csv file.
|**SAVE_AGENT_LIST.hours_to_save**| Specify the hours to save in the file. Used to reduce the size of the file.
|**COMPLIANCE_RATE**| A general compliance rate factor. Not used.
|**ZONE_INTERVENTION_INFO**| In the simulations, hypthetical zones are created using K-means. If a specific zone needs to be locked down, it can be done using this parameter. The zones to be locked can be specified in **ZONE_ID** (it is a list) and each zone will be locked if the number of active cases exceeds **ZONE_THRESHOLD_INFECTED**. The **SEQUENTIAL** parameter can be used to enable a scenario where zones are locked down only one at time.
|**NUM_ZONES**| A parameter to specify the number of zones to be created using K-means.
|**SELF_ISOLATION**| The scenario of Self isolation can be enabled by setting the **flag** parameter in this dictionary to true. In this scenario, infected individuals who get symptomatic will isolate at their residences until they recover. The **COMPLIANCE_RATE** gives the percentage of the population who complies with self isolation.
|**HOSPITALIZATION**| To enable hospitalization, i.e., individuals who get infected and get symptomatic can be hospitalized based on their age.
|**MASK_COMPLIANCE**| To enable mask compliance. **threshold** specifies the number of active cases after which we enable mask wearing. **MASK_INFECTION_FACTOR** provides the probability factors (two values| wears mask, does not wear a mask) that are used to reduce the probability of infection.
|**CONTACT_TRACING**| The parameter to enable contact tracing. **PAST_DAYS** considers the past number of days. **ISOLATION_PERIOD** specifies the period of isolation in days. **threshold_infected** specifies the number of active cases after which we enable contact tracing.
|**TESTING_INFO**| To enable the scenario of COVID-19 testing. **daily_rate** specifies the rate of daily testing. **threshold_infected** specifies the number of active cases after which we enable testting.
|**ASYMPTOMATIC_RATIO_IN_RECOVERY_MODEL**| To specify the ratio of the number of asymptomatic cases.
|**MAX_DAYS_OF_INFECTION**| the max number of days of infection. A person will recover between 12 to 30 days or die within this period.
|**INFECTION_RATE_FACTOR**| A factor to descrease the infectiousness. A higher factor corresponds to reduced infectiousness. 
|**w_shape**| A parameter for the infectiousness model (Weibull distribution)
|**w_scale**| A parameter for the infectiousness model (Weibull distribution)
|**meanlog**| A parameter for the incubation time distribution (lognormal)
|**sdlog**| A parameter for the incubation time distribution (lognormal)
|**NUMBER_ITERATIONS_PER_DAY**| Currently hardcoded as 5. 
|**get_risk**| Not used. Planned to get risks in the simulation if needed.
|**TYPE_OF_JOB.type**|Specify the type of job in the population.
|**TYPE_OF_JOB.prob**|Specify the distribution for all jobs except "No Job".
|**MORBIDITY.morbidity_infection_risk_factors**| Specify the risk factors for the comorbidities.
|**MORBIDITY.population_with_no_condition_table**| Percentage of each age group that has no comorbidity.
<footer style="padding:10px;background:#f9f9f9;border-bottom:3px solid #394851">©2020 Teradata. All Rights Reserved</footer>