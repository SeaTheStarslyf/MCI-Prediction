import os
template1 = '{age} year old. '
template2 = '{gender}. '
template4 = 'with {educ} years of education. ' 
template5 = 'The MMSE score is {mmse}. '
template6 = 'The CDR score is {cdr}. '
template7 = 'The logical memory score is {ldeltotal}. '
template8 = 'The diagnosis is {label}. '

_LABEL_MAP = {
    'CN': 'CN',
    'SMC': 'CN',
    'MCI': 'MCI',
    'EMCI': 'MCI',
    'LMCI': 'MCI',
    'AD': 'AD',
    'Dementia': 'AD',
    'Patient': 'AD'
}

mmse_cdr_dict = {}
with open('/data2/qiuhui/data/adni/MMSE.csv') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('"Phase"'):
            continue
        line_info = line.strip('\n').strip('"').split('","')
        rid,month,ymd,mmse = line_info[2],line_info[5],line_info[6],line_info[56]
        if rid not in mmse_cdr_dict.keys():
            mmse_cdr_dict[rid] = {
                month:{
                    'ymd': ymd,
                    'mmse':mmse
                }
            }
        else:
            mmse_cdr_dict[rid][month] = {
                'ymd': ymd,
                'mmse':mmse
            }

with open('/data2/qiuhui/data/adni/CDR.csv') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('"Phase"'):
            continue
        line_info = line.strip('\n').strip('"').split('","')
        rid,month,ymd,cdr = line_info[2],line_info[5],line_info[6],line_info[17]
        if rid not in mmse_cdr_dict.keys():
            mmse_cdr_dict[rid] = {
                month:{
                    'ymd': ymd,
                    'mmse': '',
                    'cdr':cdr
                }
            }
        else:
            if month not in mmse_cdr_dict[rid].keys():
                mmse_cdr_dict[rid][month] = {
                    'ymd': ymd,
                    'mmse': '',
                    'cdr':cdr
                }
            else:
                mmse_cdr_dict[rid][month]['cdr'] = cdr



with open('/data2/qiuhui/data/adni/BLCHANGE.csv') as f:
    lines = f.readlines()
    index = 0
    for line in lines:
        print(index)
        index+=1
        if line.startswith('"Phase"'):
            continue
        line_info = line.strip('\n').strip('"').split('","')
        # if line_info[0] == '':
        #     continue
        try:
            doc = line_info[25]
        except:
            import pdb;pdb.set_trace()
        if doc != '' and doc != '-4':
            rid, month = line_info[2], line_info[5]
            # import pdb;pdb.set_trace()
            if month == 'sc' or month == '':
                continue
            if month == 'bl':
                month = 'sc'
            if month not in mmse_cdr_dict[rid].keys():
                mmse_cdr_dict[rid][month] = {
                    'mmse': '',
                    'cdr': '',
                    'doc': doc
                }
            mmse_cdr_dict[rid][month]['doc'] = doc

# import pdb;pdb.set_trace()


index2 = 0
text_dict = {}
with open('/data2/qiuhui/data/adni/ADNIMERGE.csv') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('"RID"'):
            continue
        line_info = line.strip('\n').strip('"').split('","')

        rid,ptid,month,ymd,bl_age,gender,educ,mmse,ldeltotal,label = line_info[0],line_info[3],line_info[5],line_info[6],line_info[8],line_info[9],line_info[10],line_info[26],line_info[31],line_info[60]



        if bl_age != '': # 不为空
            if month == 'bl':
                age = bl_age
            else:
                age = float(bl_age) + float(month.strip('m'))/12.0
        else:
            age = bl_age

        cdr = ''
        try:
            if month =='bl':
                cdr = mmse_cdr_dict[rid]['sc']['cdr']
            else:
                cdr = mmse_cdr_dict[rid][month]['cdr']
        except:
            pass

        doc = ''
        try:
            if month =='bl':
                doc = mmse_cdr_dict[rid]['sc']['doc']
            else:
                doc = mmse_cdr_dict[rid][month]['doc']
        except:
            pass

        if age!='':
            res = template1.format(age=str(int(float(age))))
        if gender!='':
            res += template2.format(gender=gender)
        if educ !='':
            res+=template4.format(educ=educ)
        if mmse!='':
            res+=template5.format(mmse=mmse)
        if cdr!='':
            res+=template6.format(cdr=cdr)
        if ldeltotal!='':
            res+=template7.format(ldeltotal=ldeltotal)
        if doc!='':
            res+=doc+'. '
            index2+=1
            print('doc num: ',index2)

        if ptid not in text_dict.keys():
            text_dict[ptid] = {
                ymd:{
                    'text': res,
                    'label':label
                }
            }
        else:
            text_dict[ptid][ymd] = {
                'text': res,
                'label':label
            }
        
# with open('/data2/qiuhui/data/adni/demographic_label.csv') as f:
#     lines = f.readlines()
#     for line in lines:
#         if line.startswith('"Image'):
#             continue
#         line_info = line.strip('\n').strip('"').split('","')
#         ptid,label,gender,age,ymd=line_info[1],line_info[2],line_info[3],line_info[4],line_info[9]
#         if label not in ['CN', 'AD', 'MCI']:
#             continue
#         demo_m,demo_d,demo_y = ymd.split('/')
#         ymd = demo_y + '-' + demo_m + '-' + demo_d
#         if gender == 'M':
#             gender = 'Male'
#         else:
#             gender = 'Female'

#         if age!='':
#             res = template1.format(age=str(int(float(age))))
#         if gender!='':
#             res += template2.format(gender=gender)
#         rid = ptid.split('_')[-1]
#         if rid in mmse_cdr_dict:
#             for month in mmse_cdr_dict[rid].keys():
#                 cdr_y,cdr_m,cdr_d = mmse_cdr_dict[rid][month]['ymd'].split('-')
#                 demo_y,demo_m,demo_d = ymd.split('-')
#                 if cdr_y == demo_y and (int(demo_m) <= int(cdr_m)+2 and int(demo_m) >= int(cdr_m)-2):
#                     try:
#                         cdr = mmse_cdr_dict[rid][month]['cdr']
#                     except:
#                         cdr = ''
#                     mmse = mmse_cdr_dict[rid][month]['mmse']
#                     if mmse!='':
#                         res+=template5.format(mmse=mmse)
#                     if cdr!='':
#                         res+=template6.format(cdr=cdr)

#         if ptid not in text_dict.keys():
#             text_dict[ptid] = {
#                 ymd:{
#                     'text': res,
#                     'label':label
#                 }
#             }
#         else:
#             merge_ymds = text_dict[ptid].keys()
#             _match = False
#             for merge_ymd in text_dict[ptid].keys():
#                 cur_y,cur_m,cur_d = ymd.split('-')
#                 merge_y,merge_m,merge_d = merge_ymd.split('-')
#                 if merge_y == cur_y and (int(merge_m) <= int(cur_m)+2 and int(merge_m) >= int(cur_m)-2):
#                     _match = True
#                     if text_dict[ptid][merge_ymd]['label'] == '':
#                         text_dict[ptid][merge_ymd]['label'] = label
#                         print('{} {} has no label in ADNIMERGE, pick label in demographic-label.csv,{}:{}'.format(ptid,merge_ymd,ymd,label))
#             if not _match:
#                 text_dict[ptid][ymd] = {
#                     'text': res,
#                     'label':label
#                 }


to_write = []

count = 0
adni_root = '/data2/qiuhui/data/adni/images'
ptid_list = os.listdir(adni_root)
for ptid in ptid_list:
    ymd_list = os.listdir(os.path.join(adni_root,ptid))
    if '.DS_Store' in ymd_list:
        ymd_list.remove('.DS_Store')
    for ymd in ymd_list:
        count+=1
        file_path = os.path.join(adni_root,ptid,ymd,'t1.nii.gz')
        if ptid in text_dict.keys():
            for text_ymd in text_dict[ptid].keys():
                img_y,img_m,img_d = ymd.split('-')
                text_y,text_m,text_d = text_ymd.split('-')
                if img_y == text_y and (int(img_m) <= int(text_m)+2 and int(img_m) >= int(text_m)-2):
                    text = text_dict[ptid][text_ymd]['text']
                    label = text_dict[ptid][text_ymd]['label']
                    if label != '':
                        text+=template8.format(label=label)
                    to_write.append(file_path + '\t'+ text + '\n' )
                    # if label == '':
                    #     import pdb;pdb.set_trace()
                    # break


with open('/data2/qiuhui/code/MedBLIPv2/local_data/ADNI-train-4.csv','a+') as f:
    for line in to_write:
        f.write(line)




