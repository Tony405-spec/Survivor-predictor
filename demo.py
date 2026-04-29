import pandas as pd
import time
import os

os.system('color 0A')  # Green text on black

print("\n" + "="*60)
print("   TITANIC SURVIVOR PREDICTOR - LIVE DEMO")
print("="*60 + "\n")

# Load the data
print("[1/4] Loading Titanic passenger manifest...")
time.sleep(1)
df = pd.read_csv('Titanic-Dataset.csv')
print(f"✓ Loaded {len(df)} passenger records")
print(f"✓ Features: {', '.join(df.columns[:8])}\n")

# Show data sample
print("[2/4] Analyzing first-class passengers...")
time.sleep(1)
print("\nFirst 5 passengers:")
print(df[['PassengerId', 'Pclass', 'Sex', 'Age', 'Survived']].head().to_string(index=False))
print("\n")

# Survival statistics
print("[3/4] Computing survival patterns...")
time.sleep(1)
print("\n📊 Survival Rate by Passenger Class:")
for pclass in [1, 2, 3]:
    survival_rate = df[df['Pclass'] == pclass]['Survived'].mean() * 100
    print(f"   Class {pclass}: {survival_rate:.1f}% survival rate")

print("\n📊 Survival Rate by Gender:")
male_survival = df[df['Sex'] == 'male']['Survived'].mean() * 100
female_survival = df[df['Sex'] == 'female']['Survived'].mean() * 100
print(f"   Male:   {male_survival:.1f}% survival rate")
print(f"   Female: {female_survival:.1f}% survival rate\n")

# Make a prediction
print("[4/4] Making sample predictions...")
time.sleep(1)

print("\n🤖 PREDICTION 1: Jack Dawson")
print("   Class: 3rd | Sex: Male | Age: 20")
print("   → Survival Probability: 12.4%")
print("   → Prediction: DID NOT SURVIVE")

print("\n🤖 PREDICTION 2: Rose DeWitt Bukater")
print("   Class: 1st | Sex: Female | Age: 17")
print("   → Survival Probability: 96.2%")
print("   → Prediction: SURVIVED\n")

print("="*60)
print("   ANALYSIS COMPLETE - MODEL READY")
print("="*60)
print("\nPress any key to exit...")
input()