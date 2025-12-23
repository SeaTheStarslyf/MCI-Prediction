import pandas as pd
import os

gender_map = {
    '1': 'Male',
    '2': 'Female'
}

hand_map = {
    '1': 'Left',
    '2': 'Right'
}

label_map = {
    '1': 'CN',
    '2': 'Impaired-not-MCI',
    '3': 'MCI',
    '4': 'Dementia'
}

template1 = '{age} year old. '
template2 = '{gender}. '
template3 = '{hand} handed. '
template4 = 'with {educ} years of education. ' 
template5 = 'The MMSE score is {mmse}. '
template6 = 'The CDR score is {cdr}. '
template7 = 'The logical memory score is {ldeltotal}. '
template8 = 'The diagnosis is {label}. '



total_dict = {}
with open('/data2/qiuhui/data/nacc/investigator_ftldlbd_nacc61.csv') as f:
    lines = f.readlines()
    for line in lines:
        line_info = line.strip('\n').split(',')
        if line_info[0] =='NACCID': # 第一行
            heads = line_info
            continue
        else:
            naccid = line_info[heads.index('NACCID')]
            mmse = line_info[heads.index('NACCMMSE')] # 0-30
            cdr = line_info[heads.index('CDRSUM')] # 0-18
            logical_memory = line_info[heads.index('MEMUNITS')] # 0–25
            gender = line_info[heads.index('SEX')] # 1:male 2:female
            educ = line_info[heads.index('EDUC')] # 0-36
            hand = line_info[heads.index('HANDED')] # 1:左手 2右手 3双手不全 9未知
            birth_year = line_info[heads.index('BIRTHYR')]
            label = line_info[heads.index('NACCUDSD')] # 1 = Normal cognition 2 = Impaired-not-MCI 3 = MCI 4 = Dementia
            visit_year = line_info[heads.index('VISITYR')]
            visit_num = line_info[heads.index('NACCVNUM')] 

        if naccid not in total_dict.keys():
            total_dict[naccid] = [
                {
                    'visit_num': visit_num,
                    'visit_year': visit_year,
                    'mmse': mmse,
                    'cdr': cdr,
                    'logical_memory': logical_memory,
                    'gender': gender,
                    'hand': hand,
                    'educ': educ,
                    'birth_year': birth_year,
                    'label': label,
                }
            ]
        else:
            total_dict[naccid].append({
                'visit_num': visit_num,
                'visit_year': visit_year,
                'mmse': mmse,
                'cdr': cdr,
                'logical_memory': logical_memory,
                'gender': gender,
                'hand': hand,
                'educ': educ,
                'birth_year': birth_year,
                'label': label,
            })

mri_dict = {}
with open('/data2/qiuhui/data/nacc/investigator_mri_nacc61.csv') as f:
    lines = f.readlines()
    for line in lines:
        line_info = line.strip('\n').split(',')
        if line_info[0] =='NACCADC': # 第一行
            heads = line_info
            continue
        else:
            naccid = line_info[heads.index('NACCID')]
            mri_year = line_info[heads.index('MRIYR')]
            zip_name = line_info[heads.index('NACCMRFI')]
            mri_num = line_info[heads.index('NACCMNUM')]

        mri_dict[zip_name] = {
            'naccid': naccid,
            'mri_year': mri_year,
            'mri_num': mri_num
        }
        

from pathlib import Path
image_root = '/data2/qiuhui/data/nacc/images'
zip_files = os.listdir(image_root)
cnt=0
for zip_file in zip_files:
    img_dir = os.path.join(image_root,zip_file)
    for file_path in sorted(Path(img_dir).rglob('*.nii')):
        cnt+=1
        print(cnt,str(file_path))
        if 'ni' in zip_file:
            new_zip_file = zip_file[:-2]+'.zip'
        else:
            import pdb;pdb.set_trace()
        if new_zip_file not in mri_dict.keys():
            print(new_zip_file, 'not in mri csv, pass!')
            continue
        naccid = mri_dict[new_zip_file]['naccid']
        mri_year = mri_dict[new_zip_file]['mri_year']
        mri_num = mri_dict[new_zip_file]['mri_num']
        

        if naccid not in total_dict.keys():
            import pdb;pdb.set_trace()
        gender = total_dict[naccid][0]['gender']
        hand = total_dict[naccid][0]['hand']
        birth_year = total_dict[naccid][0]['birth_year']
        age = int(mri_year) - int(birth_year)

        match = False
        for item in total_dict[naccid]:
            if item['visit_year'] == mri_year:
                match = True
                mmse = item['mmse']
                cdr = item['cdr']
                logical_memory = item['logical_memory']
                educ = item['educ']
                label = item['label']
                break

        if not match:
            for item in total_dict[naccid]:
                if (int(item['visit_year']) <= int(mri_year)+1) or (int(item['visit_year']) >= int(mri_year)-1):
                    match = True
                    mmse = item['mmse']
                    cdr = item['cdr']
                    logical_memory = item['logical_memory']
                    educ = item['educ']
                    label = item['label']
                    break
        if not match:
            import pdb;pdb.set_trace()

        
        res = template1.format(age=str(int(float(age))))
        res += template2.format(gender=gender_map[gender])
        if hand in ['1','2']:
            res += template3.format(hand=hand_map[hand])

        if int(educ) <=36 and int(educ) >=0:
            res+=template4.format(educ=educ)
        if int(mmse) <=30 and int(mmse) >=0:
            res+=template5.format(mmse=mmse)
        if float(cdr) <=18 and float(cdr) >=0:
            res+=template6.format(cdr=cdr)
        if int(logical_memory) <=25 and int(logical_memory) >=0:
            res+=template7.format(ldeltotal=logical_memory)
        if label in ['1','2','3','4']:
            res+=template8.format(label=label_map[label])

        with open('/data2/qiuhui/code/MedBLIPv2/local_data/NACC-2.csv','a+') as f:
            line = str(file_path) + '\t' + res + '\n'
            f.write(line)








