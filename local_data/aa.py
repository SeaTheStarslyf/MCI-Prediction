import os 

with open('/data-pool/data/data2/qiuhui/code/Alifuse_bibm/local_data/aaa.txt') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('rj'):
            # import pdb;pdb.set_trace()
            items = line.strip().split('\t')
            try:
                rid, label, mmse, age, edu_year, gender = items[0],items[3],items[4],items[15],items[16],items[17]
            except:
                import pdb;pdb.set_trace()

            file_dir = os.path.join('/data-pool/data/data2/qiuhui/data/rj_AD',rid)
            
            if not os.path.exists(file_dir):
                continue
            sub_file_name = os.listdir(file_dir)[0]
            file_path = os.path.join(file_dir,sub_file_name)

            text = '{} years old. '.format(age)
            gender_map={
                '1': 'Male',
                '0': 'Female'
            }
            text+= '{}. '.format(gender_map[gender])
            # text+='with {} years of education. '.format(edu_year)
            text+='The MMSE score is {}. '.format(mmse)
            label_map = {
                '0': 'CN',
                '0.5': 'MCI',
                '1': 'MCI',
                '2': 'AD'
            }
            text+='The diagnosis is {}. '.format(label_map[label])
            

            # import pdb;pdb.set_trace()
            with open('/data-pool/data/data2/qiuhui/code/Alifuse_bibm/local_data/RENJI2.csv','a+') as f2:
                f2.write(file_path+'\t'+text+'\n')