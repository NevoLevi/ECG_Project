import pandas as pd

df = pd.read_csv('frontend/public/cat_net_classification_report_fixed.csv', index_col=0)

print("Looking for patients with 2+ abnormal classifications in any 5-window group...")
print("=" * 70)

trigger_patients = []

for patient_id in df.index:
    classifications = df.loc[patient_id].values
    
    # Check every 5-window group (windows processed every 5 steps)
    for start_window in range(0, len(classifications)-4, 5):
        window_group = classifications[start_window:start_window+5]
        abnormal_count = sum(1 for c in window_group if c != 'N')
        
        if abnormal_count >= 2:
            abnormal_types = [c for c in window_group if c != 'N']
            # Find majority abnormal type
            type_counts = {}
            for t in abnormal_types:
                type_counts[t] = type_counts.get(t, 0) + 1
            majority_type = max(type_counts.items(), key=lambda x: x[1])[0]
            
            print(f"🚨 TRIGGER: Patient {patient_id}")
            print(f"   Window {start_window+1}-{start_window+5}: {list(window_group)}")
            print(f"   Abnormal count: {abnormal_count}/5")
            print(f"   Majority type: {majority_type}")
            print(f"   Abnormal positions in window: {[i+1 for i, c in enumerate(window_group) if c != 'N']}")
            print()
            
            trigger_patients.append({
                'patient': patient_id,
                'window_start': start_window + 1,
                'window_group': list(window_group),
                'abnormal_count': abnormal_count,
                'majority_type': majority_type
            })
            break  # Only show first trigger for each patient

print(f"Total patients that should trigger abnormal status: {len(trigger_patients)}")

# Also show first few windows for a few patients to verify data structure
print("\n" + "=" * 70)
print("Sample data verification:")
for patient_id in [1, 2, 5, 10]:
    first_20 = list(df.loc[patient_id].values[:20])
    print(f"Patient {patient_id} windows 1-20: {first_20}") 