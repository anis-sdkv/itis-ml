import pandas as pd
import numpy as np
from sympy import pprint



def main():
    symptoms = pd.read_csv('datasets/symptom.csv', sep=";")
    diseases = pd.read_csv('datasets/disease.csv', sep=";")

    diseases['P(disease)'] = diseases['количество пациентов'] / diseases['количество пациентов'].sum()

    # TODO доделать
    # find P(symphom|disease)
    for symptom_name in symptoms:
        for disease_name in diseases['disease']:
            # symptoms[f'P{i}|{}']
            print(disease_name, diseases[disease_name])#['disease'])
            break
        break

main()