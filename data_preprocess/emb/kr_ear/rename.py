import os
import re

csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
# pattern = re.compile(r'(\d+)_(\d+)hop_attwSTF(\d+)\.csv')
# pattern = re.compile(r'(\d+)_(\d+)hop_att_rel.csv')
# pattern = re.compile(r'(\d+)_(\d+)hop_att_relST(\d+).csv')
# pattern = re.compile(r'(\d+)_(\d+)hop_att_relSTF(\d+).csv')
# pattern = re.compile(r'(\d+)_(\d+)hop_att_relwSTF(\d+).csv')
# pattern = re.compile(r'(\d+)_(\d+)hop_attwSTF(\d+)\.csv')
# pattern = re.compile(r'ic_(\d+)_(\d+)hop_att_relwSTF(\d+).csv')
pattern = re.compile(r'poi_S_(\d+)d')
# {}d_{}hop_poi_S.csv
for old_csv_file in csv_files:
    match = pattern.match(old_csv_file)
    if match:
        d, h = match.groups()
        # new_csv_file = 'poi_S_{}d_{}hop.csv'.format(d, h)
        # new_csv_file = 'poi_ST_{}d_{}hop_s{}'
        # new_csv_file = 'poi_STF_{}d_{}hop_s{}.csv'.format(d, h, s)
        new_csv_file = 'S_{}d_{}hop.csv'.format(d, h)
        print(old_csv_file, "=> d =", d, ", h =", h, "new_csv_file=", new_csv_file)
        # exit(0)
        # 执行重命名
        os.rename(old_csv_file, new_csv_file)
        print(f"{old_csv_file}  =>  {new_csv_file}")
        # exit(0)


# '{}_{}hop_attwSTF{}.csv'
# '{}d_{}hop_s{}_poi.csv'
# old_file_name = ''
# new_file_name = ''


# print(csv_files)