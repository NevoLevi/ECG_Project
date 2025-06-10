import pandas as pd

df = pd.read_csv('frontend/public/cat_net_classification_report_fixed.csv', index_col=0)
print('Shape:', df.shape)
print('Patients:', list(df.index[:10]))
print('Heartbeats:', list(df.columns[:10]))
print()
print('Patient 2 first 20 windows:', list(df.loc[2].values[:20]))
print('Patient 10 first 20 windows:', list(df.loc[10].values[:20]))
print()

# Find some patients with abnormalities
for patient in [1, 2, 3, 4, 5, 10, 15, 20]:
    classifications = df.loc[patient].values
    abnormal_count = sum(1 for c in classifications if c != 'N')
    abnormal_windows = [i+1 for i, c in enumerate(classifications) if c != 'N']
    if abnormal_count > 0:
        print(f'Patient {patient}: {abnormal_count} abnormal windows at positions {abnormal_windows[:10]}')
        # Show a sample 5-window that might trigger
        for start in range(0, min(50, len(classifications)-4), 5):
            window = classifications[start:start+5]
            abnormal_in_window = sum(1 for c in window if c != 'N')
            if abnormal_in_window >= 2:
                print(f'  -> Windows {start+1}-{start+5}: {list(window)} ({abnormal_in_window}/5 abnormal)')
                break 