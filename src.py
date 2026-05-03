import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


COLUMNS = ['age','sex','cp','trestbps','chol','fbs','restecg',
           'thalach','exang','oldpeak','slope','ca','thal','target']

print("STEP 1 - Loading File")
print("-" * 40)

try:
    df = pd.read_csv(r'C:\Users\arvap\Downloads\Python Project\heart.csv',
                     header=None, names=COLUMNS, na_values='?')
    print("File loaded successfully!")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
except FileNotFoundError:
    print("Error: File not found!")
    exit()
except PermissionError:
    print("Error: No permission to read file!")
except ValueError:
    print("Error: Wrong data type in file!")
except Exception as e:
    print("Some other error:", e)

print("\nSTEP 2 - Data Exploration")
print("-" * 40)

print("\nFirst 10 rows:")
print(df.head(3))

print("\nLast 5 rows:")
print(df.tail(5))

print("\nDataFrame Info:")
print(df.info())

print("\nPatients older than 60 OR cholesterol above 240:")
count = 0
for i in range(len(df)):
    age = df['age'][i]
    chol = df['chol'][i]
    if age > 60 or chol > 240:
        print("  Index:", i, "| Age:", age, "| Chol:", chol)
        count += 1
        if count == 8:
            print("  ... and more patients")
            break

print("\nSTEP 3 - Missing Values")
print("-" * 40)
print("Missing values in each column:")
print(df.isnull().sum())

for col in ['ca', 'thal']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    col_mean = df[col].mean()
    missing = df[col].isnull().sum()
    if missing > 0:
        df[col] = df[col].fillna(round(col_mean, 2))
        print("Filled", col, "- missing:", missing, "- mean:", round(col_mean, 2))

print("Missing after fix:", df.isnull().sum().sum())


print("\nSTEP 4 - New Columns")
print("-" * 40)
df['RiskScore'] = (df['age'] * 0.3) + (df['chol'] * 0.4) + (df['trestbps'] * 0.3)
ratio_list = []
for i in range(len(df)):
    try:
        if df['chol'][i] == 0:
            raise ZeroDivisionError
        ratio = df['trestbps'][i] / df['chol'][i]
        ratio_list.append(round(ratio, 4))
    except ZeroDivisionError:
        print("Warning: chol=0 at row", i, "using 0")
        ratio_list.append(0)
df['BP_Chol_Ratio'] = ratio_list

age_groups = []
for i in range(len(df)):
    age = df['age'][i]
    if age < 40:
        age_groups.append('Young')
    elif age <= 60:
        age_groups.append('Middle')
    else:
        age_groups.append('Senior')
df['AgeGroup'] = age_groups

print("AgeGroup counts:")
print(df['AgeGroup'].value_counts())

print("\nSTEP 5 - Age Group Analysis")
print("-" * 40)

groups = ['Young', 'Middle', 'Senior']

for grp in groups:
    grp_df = df[df['AgeGroup'] == grp]
    print("\nGroup:", grp)
    print("  Avg RiskScore:", round(grp_df['RiskScore'].mean(), 2))
    print("  Avg Cholesterol:", round(grp_df['chol'].mean(), 2))
    print("  Avg Heart Rate:", round(grp_df['thalach'].mean(), 2))


print("\nSTEP 6 - Patient Dictionary")
print("-" * 40)

patient_dict = {}
for i in range(len(df)):
    patient_dict[i] = {
        'age'      : df['age'][i],
        'chol'     : df['chol'][i],
        'trestbps' : df['trestbps'][i],
        'RiskScore': round(df['RiskScore'][i], 2),
        'target'   : df['target'][i]
    }

print("Sample - Patient #5:")
for key in patient_dict[5]:
    print(" ", key, ":", patient_dict[5][key])

print("\nSTEP 7 - Disease Rate per Age Group")
print("-" * 40)

for grp in groups:
    grp_df = df[df['AgeGroup'] == grp]
    total = len(grp_df)
    diseased = (grp_df['target'] != 0).sum()
    try:
        if total == 0:
            raise ZeroDivisionError
        rate = diseased / total * 100
        print(grp, "disease rate:", round(rate, 2), "%")
    except ZeroDivisionError:
        print("Warning: No patients in group", grp, "- using 0")


print("\nSTEP 8 - Statistics")
print("-" * 40)

cols = ['age', 'trestbps', 'chol', 'thalach', 'RiskScore']

print("\nManual Statistics (using loops):")
for col in cols:
    values = list(df[col])
    total = 0
    for v in values:
        total = total + v
    mean = total / len(values)

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 0:
        median = (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
    else:
        median = sorted_vals[n//2]

    print(col, "-> mean:", round(mean, 2),
          "| median:", round(median, 2),
          "| min:", sorted_vals[0],
          "| max:", sorted_vals[-1])

print("\nPandas describe():")
print(df[cols].describe().round(2))

print("\nAvg Cholesterol grouped by target:")
print(df.groupby('target')['chol'].mean().round(2))


print("\nCreating charts...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Heart Disease Analysis', fontsize=14, fontweight='bold')

# Chart 1 - Disease count
counts = [(df['target'] == 0).sum(), (df['target'] != 0).sum()]
axes[0, 0].bar(['No Disease', 'Disease'], counts, color=['steelblue', 'tomato'])
axes[0, 0].set_title('Disease vs No Disease')
axes[0, 0].set_ylabel('Count')

# Chart 2 - Age histogram
axes[0, 1].hist(df['age'], bins=15, color='steelblue', edgecolor='black')
axes[0, 1].set_title('Age Distribution')
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Count')

# Chart 3 - Avg RiskScore by Age Group
avg_scores = [df[df['AgeGroup'] == g]['RiskScore'].mean() for g in groups]
axes[1, 0].bar(groups, avg_scores, color=['green', 'orange', 'purple'])
axes[1, 0].set_title('Avg RiskScore by Age Group')
axes[1, 0].set_ylabel('RiskScore')

# Chart 4 - Cholesterol by Disease
axes[1, 1].hist(df[df['target']==0]['chol'], bins=15, alpha=0.7, label='No Disease', color='steelblue')
axes[1, 1].hist(df[df['target']!=0]['chol'], bins=15, alpha=0.7, label='Disease', color='tomato')
axes[1, 1].set_title('Cholesterol Distribution')
axes[1, 1].set_xlabel('Cholesterol')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('heart_disease_charts.png', dpi=120)
plt.close()
print("Charts saved!")


print("\nSTEP 9 - Saving Report")
print("-" * 40)

try:
    file = open('heart_disease_analysis_report.txt', 'w')

    file.write("HEART DISEASE CLEVELAND - ANALYSIS REPORT\n")
    file.write("=" * 50 + "\n\n")

    file.write("Total Patients: " + str(len(df)) + "\n")
    file.write("Disease Cases:  " + str((df['target'] != 0).sum()) + "\n")
    file.write("No Disease:     " + str((df['target'] == 0).sum()) + "\n\n")

    file.write("AGE GROUP DISTRIBUTION\n")
    file.write("-" * 30 + "\n")
    for grp in groups:
        count = len(df[df['AgeGroup'] == grp])
        file.write(grp + ": " + str(count) + " patients\n")

    file.write("\nAVERAGE STATS BY AGE GROUP\n")
    file.write("-" * 30 + "\n")
    for grp in groups:
        g = df[df['AgeGroup'] == grp]
        file.write(grp + " -> Chol: " + str(round(g['chol'].mean(), 1))
                   + " | RiskScore: " + str(round(g['RiskScore'].mean(), 1)) + "\n")

    file.write("\nDISEASE RATE BY AGE GROUP\n")
    file.write("-" * 30 + "\n")
    for grp in groups:
        g = df[df['AgeGroup'] == grp]
        try:
            rate = (g['target'] != 0).sum() / len(g) * 100
        except ZeroDivisionError:
            rate = 0
        file.write(grp + ": " + str(round(rate, 1)) + "%\n")

    file.write("\nOVERALL STATISTICS\n")
    file.write("-" * 30 + "\n")
    file.write("Avg Age:         " + str(round(df['age'].mean(), 1)) + "\n")
    file.write("Avg Cholesterol: " + str(round(df['chol'].mean(), 1)) + "\n")
    file.write("Avg Heart Rate:  " + str(round(df['thalach'].mean(), 1)) + "\n")
    file.write("Avg RiskScore:   " + str(round(df['RiskScore'].mean(), 1)) + "\n")

    file.write("\n" + "=" * 50 + "\n")
    file.write("END OF REPORT\n")
    file.close()
    print("Report saved successfully!")

except PermissionError:
    print("Error: Cannot write file!")
except Exception as e:
    print("Error saving report:", e)

print("\nANALYSIS COMPLETE!")