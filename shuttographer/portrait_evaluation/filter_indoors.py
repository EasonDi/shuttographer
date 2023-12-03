import os
import shutil
import pandas

COPY_IMAGES = False
MAKE_SCORES_DOC = True


if __name__=="__main__":

    if COPY_IMAGES:
        image_folder = 'Dataset/Overall/'
        indoor_folder = 'Dataset/indoor/'
        for image in os.listdir(image_folder):
            if 'Indoor' in image:
                shutil.copy(image_folder+image, indoor_folder+image)
    
    if MAKE_SCORES_DOC:
        image_folder = 'Dataset/indoor/'
        image_doc = 'Dataset/Scores_Overall.csv'
        df = pandas.read_csv(image_doc)
        data = {'image_name': [], 'quality':[]}
        for image in os.listdir(image_folder):
            data['image_name'].append(image)
            data['quality'].append(df.loc[df['IMAGE PATH']==('Overall\\'+image)]['QUALITY LEVEL'].iloc[0])
        df_save = pandas.DataFrame(data)
        df_save.to_csv('Dataset/indoor_quality_scores.csv', columns=['image_name', 'quality'], index=False)



