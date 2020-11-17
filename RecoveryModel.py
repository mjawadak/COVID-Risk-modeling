import pandas as pd

##### death prediction model
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

class RecoveryModel():

	def __init__(self,MAX_DAYS_OF_INFECTION,NUMBER_ITERATIONS_PER_DAY):

		# Loads the recovery model data and builds a decision tree model with fixed parameters.
		# We can test other models as well, but to speed up the code, we rely on a simple model.
		# The model takes as input the age and gender and predicts whether the individual recovers or not.
		# The training data (recovery_model_data.csv) is obtained from https://c3.ai/covid-19-api-documentation/#tag/LineListRecord (nCoV2019 Data Working Group and MOBS Lab)

		df = pd.read_csv("data/recovery_model_data.csv")
		df["gender"]= df["gender"].str.lower()
		df["status"]= df["status"].str.lower()
		self.le_gender = preprocessing.LabelEncoder()
		self.le_gender.fit(df["gender"].unique())
		self.le_death = preprocessing.LabelEncoder()
		self.le_death.fit(df["status"].unique())
		self.MAX_DAYS_OF_INFECTION = MAX_DAYS_OF_INFECTION
		self.NUMBER_ITERATIONS_PER_DAY = NUMBER_ITERATIONS_PER_DAY
		df["gender_int"]= self.le_gender.transform(df["gender"])
		df["status_int"]= self.le_death.transform(df["status"])

		# Train the ML model
		self.clf = DecisionTreeClassifier(min_samples_leaf=25, max_depth=3)
		self.clf.fit(df[["age","gender_int"]].values,df["status_int"].values)

	def predictDeathProbs(self,df):
		inputs = df
		death_probabilities = self.clf.predict_proba(inputs)[:,0] # a list of death probabilites for each infected individual

		# Below, dividing each prob with the average total number of infected days (15+30)/2 and then by NUMBER_ITERATIONS_PER_DAY.
		# This is because this function is called in every hourly interval, so we equally divide the probability by the average duration of the infection.
		return death_probabilities/((12+self.MAX_DAYS_OF_INFECTION)/2.0)/self.NUMBER_ITERATIONS_PER_DAY

################## TESTING CODE ##################

if __name__ == "__main__":
	recovery_model = RecoveryModel(30,24)
	print(recovery_model.predictDeathProbs([[0,1],
													[10, 1],
													[20, 1],
													[30, 1],
													[40, 1],
													[50, 1],
													[60, 1],
													[70, 1],
													[80, 1],
													[90, 1],
													]))