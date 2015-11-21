#__author__ = 'rajranj'

from __future__ import division
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.externals import joblib
from multiprocessing import Pool
from pdb import set_trace as trace
from datetime import datetime
import re
import nltk.corpus
import nltk.tokenize.punkt
import nltk.stem.snowball
import string
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer


snowball_stemmer = SnowballStemmer('english')
tokenizer = nltk.tokenize.punkt.PunktWordTokenizer()
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')


fields_to_consider = ['Companies', 'Industry', 'Title']

cent_dict = {}
c_df = pd.DataFrame()
title_dict = {}
#global title_dict

centroid = [['Founder', 'Cofounder', 'Owner'], ['Chairman'], ['President', 'VP', 'EVP', 'SVP', 'AVP'], ['Director', 'Board Of Director', 'Board of Governor', 'Board of Trustee'], ['Chief Executive Officer', 'CEO', 'COO'],
            ['Chief Technical Officer', 'CTO'], ['Chief Information Office', 'CIO'], ['Chief '], ['Dean'], ['Attorney'],
            ['Auditor'], ['Manager', 'Leader'], ['Researcher', 'Scientist'], ['Administrator', 'Admin'], ['Industrialist'],
            ['Head'], ['Marketing'], ['Advisor'], ['Analyst'], ['Software', 'Developer'],
            ['Programmer'], ['Accountant'], ['Engineer'], ['Staff'], ['Strategist'],
            ['Officer'], ['Human Resource', 'HR', 'Recruiter'], ['Intern'], ['Supervisors'],
            ['Executive'], ['Controllers'], ['Nurse'], ['Technician'], ['Computing'],
            ['Professional', 'Specialist'], ['QA', 'Quality Controller'], ['Operator'], ['Technologist'], ['Producer'],
            ['Architect'], ['Designer'], ['Client'], ['Business'], ['Manufacturer'],
            ['Services'], ['Animator'], ['Relation Manager'],

            ['Driver'], ['Player'], ['Speaker'], ['Cook'],
            ['Artist', 'Performer', 'Dancer', 'Film Maker'], ['Musician'],  ['Photographer'], ['News', 'Media', 'Journalist', 'Campaigner'],
            ['Librarian'],
            ['Clerk'],
            ['Communicator'], ['Coordinator'], ['Integrator'], ['Distributor'],
            ['Author', 'Writer', 'Blogger', 'Editor'], ['Freelancer'], ['Contributor'], ['Contractor'],
            ['Faculty', 'Tutor', 'Teacher', 'Professor', 'Educator', 'Mentor', 'Instructor', 'Lecturer'], ['Coach', 'Trainer'], ['Consultant', 'Counselor'],
            ['Publisher'], ['Broker'], ['Organizer', 'Planner'], ['Ambassador'],
            ['Investor'], ['Promoter'], ['Banker'], ['Doctor', 'Cardiologist', 'Gynecologist', 'Dentist'], ['Pharmaceutical'],
            ['Government'], ['Prosecutor'], ['Student'], ['Candidate'],
            ['Volunteer', 'self'], ['Environmentalist']
        ]

Manufacturing_Cent_List = [i for i in range(47)]
Media_Cent_List = [i for i in range(47)]+[151, 152, 153, 154, 161, 168]
Software_Cent_List = [i for i in range(47)]
Food_Cent_List = [i for i in range(47)]
Banking_Cent_List = [i for i in range(47)]
Services_Cent_List = [i for i in range(47)]
Transportation_Cent_List = [i for i in range(47)]
Education_Cent_List = [3, 8, 155, 165, 179]
Health_Cent_List = [31, 167, 175]
Sport_Cent_List = [148, 166]

List_Of_States = [['alabama', 'al'], ['alaska', 'ak'], ['arizona', 'az'], ['arkansas', 'ar'], ['california', 'ca'], ['colorado', 'co'], ['connecticut', 'ct'], ['delaware', 'de'],
                  ['florida', 'fl'], ['georgia', 'ga'], ['hawaii', 'hi'], ['idaho', 'id'], ['illinois', 'il'], ['iowa', 'ia'], ['indiana', 'in'], ['kansas', 'ks'], ['kentucky', 'ky'],
                  ['louisiana', 'la'], ['maine', 'me'], ['maryland', 'md'], ['massachusetts', 'ma'], ['michigan', 'mi'], ['minnesota', 'mn'], ['mississippi', 'ms'], ['missouri', 'mo'],
                  ['montana', 'mt'], ['nebraska', 'ne'], ['nevada', 'nv'], ['new hampshire', 'nh'], ['new jersey', 'nj'], ['new mexico', 'nm'], ['new york', 'ny'], ['north carolina', 'nc', 'silicon valley'],
                  ['north dakota', 'nd'], ['ohio', 'oh'], ['oklahoma', 'ok'], ['oregon', 'or'], ['pennsylvania', 'pa', 'philadelphia'], ['south carolina', 'sc'], ['south dakota', 'sd'], ['tennessee', 'tn'],
                  ['texas', 'tx'], ['utah', 'ut'], ['vermont', 'vt'], ['virginia', 'va'], ['west virginia', 'wv', 'West Virginia'], ['washington', 'Washington dc', 'wa'], ['wisconsin', 'wi'],
                  ['wyoming', 'wy']
                ]

List_Of_Countries = ['united states', 'us', 'usa']
Full_Title_List = []

for i in range(len(centroid)):
    if i < 47:
        cent_dict[i]=centroid[i]
    else:
        cent_dict[i+100]=centroid[i]


def dist(arg1, arg2):
    arg1 = str(arg1).lower()
    arg2 = str(arg2).lower()
    return fuzz.ratio(arg1, arg2)

def get_title_cent(title):
    d = 0
    cent = 0
    for key in cent_dict.keys():
        x = title.split(' ')
        for dict_item in cent_dict.get(key):
            if dict_item.lower() in x:
                return key
    for key in cent_dict.keys():
        for item in cent_dict.get(key):
            dst = dist(title, item)
            if d < dst:
                d = dst
                cent = key
                if d == 100:
                    return key
    if d < 77:
        for item in cent_dict.get(cent):
            if fuzz.partial_ratio(title, str(item).lower()) >= 80:
                return cent
        return 1000

    return cent


def mod_dist(a, b):
    if a > b:
        return a-b
    else:
        return b-a

#def get_title_cent(title):
#    return level_dist(title)

def get_loc_score(a, b):
    if a.lower()=='not in us' or b.lower() == 'not in us':
        return 0
    elif a.lower() == b.lower():
        return 100
    else:
        return 50

def fin_score(a, b, c, d, e):
    if c == 0:
        c = 0.01
    return (a*0.30 + b*0 + (1/c)*0.35 + d*0.20 + e*0.15)

def clean_title(title):
    title = re.sub(r'[.,:;^_${}()!~=+*\-\|?"/\'\[\]^#@&0123456789]', ' ', title)
    title = re.sub(' +',' ',title)
    return title.lower()

def clean_tweets(title):
    lst = []
    if title in ([], ['']):
        return lst
    for tweet in title:
        p = re.compile(r'<.*?>')
        tweet = p.sub('\n', tweet)
        tweet = re.sub(r'https?://\S+', ' ', tweet, flags=re.MULTILINE)
        for item in ('&lt;', '&gt;', '&amp;', '&quot;', '&', '$', '@', '!', '#', '%', '>', '<'):
            tweet = tweet.replace(item, ' ')
        tweet = re.sub(r'[.,:;^_${}()!~=+*\-\|?"/\'\[\]]', '', tweet)
        tweet = re.sub(' +', ' ', tweet)
        lst.append(tweet.lower())
    return lst

def clean_fun(title):
    lst = []
    if list(set(title)) in ([], ['']):
        return lst
    for item in title:
        lst.append(clean_tweets(item))
    return lst


def get_location(location):
    for loc in location:
        for loc_value in loc.values():
            for lst in re.split(r'[,;/]+', loc_value):
                for state in List_Of_States:
                    if lst.strip().lower() in state:
                        return state[0].title()
    for loc in location:
        for loc_value in loc.values():
            for lst in re.split(r'[,;/]+', loc_value):
                if lst.strip().lower() in List_Of_Countries:
                    return List_Of_Countries[0].title()
    return 'Not In US'


def cmp_list(lst1, lst2):
    if lst1=='' or lst2=='':
        return 0
    lst1 = map(lambda x: x.lower(), lst1)
    lst2 = map(lambda x: x.lower(), lst2)
    cmp_per = 100 * (len(list(set(lst1) & set(lst2))) / len(list(set(lst1))))
    return cmp_per


int_lst=[]

def get_interest_list(interest):
    for term in interest:
        term = term.encode('ascii', 'ignore')
        int_lst.append(term)

def stem_interest(term):
    item = []
    for token in tokenizer.tokenize(term):
        #token = lemmatizer.lemmatize(token)
        item.append(snowball_stemmer.stem(token))
    return ' '.join(item)

def get_interest_score(int_one, int_two):
    if int_one == '' or int_two == '':
        return 0
    return (100 * (len(list(set(int_one) & set(int_two))) / len(list(set(int_two)))))


def get_normalized_interest(interest):
    if interest == '':
        return interest
    interest = clean_tweets(interest)
    item=[]
    for term in interest:
        #term = lemmatizer.lemmatize(term)
        item.append(snowball_stemmer.stem(term))
    return item



def get_cluster_key(titles):
    lst = []
    if titles == '':
        lst.append(1000)
    elif list(set(titles)) in ([], ['']):
        lst.append(1000)
    else:
        for title in titles:
            if title == '':
                lst.append(1000)
            elif list(set(title)) in ([], ['']):
                lst.append(1000)
            else:
                lst.append(title_dict.get(title)[0])
    mn = min(lst)
    return [mn, lst.index(mn)]

def get_company_score(cmpy_one, cmpy_two):
    if cmpy_one == '' or cmpy_two == '':
        return 0
    return (100 * (len(list(set(cmpy_one) & set(cmpy_two))) / len(list(set(cmpy_two)))))

def get_title_list(titles):
    if titles == '':
        return ''
    elif list(set(titles)) in ([], ['']):
        return ''
    else:
        for title in titles:
            if title == '':
                continue
            elif list(set(title)) in ([],['']):
                continue
            else:
                Full_Title_List.append(title)

def recomendLead(selected_lead = None):
    """
    This method will calculate the match score of the selected lead to the dataset leads in the following manner
    1. Calculate the match score of each of the selected lead companies with each of the other leads companies and keep the best match score
    2. Repeat the above step for Industry and Title
    3. Finally calculate the Final match score based on the match scores of the individual scores
    """
    now = datetime.now()
    #"""
    #dataset = joblib.load('dataset_test.joblib')

    #dataset = joblib.load("dataset_dev.joblib")
    dataset = joblib.load('dataset_pro.joblib')

    print 'Data Loading Completed'
    
    #dataset=dataset[0:100000]
    print 'Data Set Size', dataset.shape
    dataset = dataset.fillna('')

    
    print 'Clustering Started'
    title_df = []
    #p=Pool(processes=8)
    #title_df = p.map(get_title_list, dataset.Titles)
    #p.close()
    #p.join()
    #trace()
    for index, row in dataset.iterrows():
        titles = row['Titles']
        if titles == '':
            continue
        elif list(set(titles)) in ([], ['']):
            continue
        else:
            for title in titles:
                if title=='':
                    continue
                title_df.append(title)

    title_df=list(set(title_df))
    title_df = pd.DataFrame(title_df, columns=['Title'])

    print 'Title Listing Completed'
    #lst=[]
    p= Pool(processes=8)
    title_df['Cleaned_Title'] = p.map(clean_title, title_df.Title)
    p.close()
    p.join()

    print 'Titles Cleaning Completed', 'Title Clustering Started'

    p = Pool(processes=8)
    title_df['Clustered_Title'] = p.map(get_title_cent, title_df.Cleaned_Title)
    p.close()
    p.join()

    print 'Title Clustering Completed'
    x = title_df[['Title', 'Clustered_Title']]
    global title_dict
    title_dict = {k: g["Clustered_Title"].tolist() for k, g in x.groupby("Title")}

    print 'Title Filling in Main Dataset'
    dataset['Titles_Level'] = dataset.Titles.apply(get_cluster_key)
    print 'Clustering Completed'
    print datetime.now()- now


    #                       Location Clustering
    print 'Location Clustering Started'
    p=Pool(processes=8)
    dataset['Location'] = p.map(get_location, dataset.locations)
    p.close()
    p.join()
    print 'Location Clustering Completed'


    #                        Interest Normalization
    print 'Interest Normalization Started'
    p=Pool(processes=8)
    dataset['Normalized_Interest'] = p.map(get_normalized_interest, dataset.interests)
    p.close()
    p.join()
    print 'Interest Normalization Completed'
    #"""
    #dataset = joblib.load('prepared_dev_dataset.joblib')
    #print dataset.shape
    #trace()
    print datetime.now() - now
    title_level_list = []
    company_list = []
    title_list = []
    for index, row in dataset.iterrows():
        tmp = row['Titles_Level']
        title_level_list.append(tmp[0])
        if tmp[0]!=1000:
            title_list.append(row['Titles'][tmp[1]])
        else:
            title_list.append(row['Titles'])
        if isinstance(row['Companies'], list):
            try:
                company_list.append(row['Companies'][tmp[1]])
            except:
                company_list.append('')
        else:
            company_list.append('')
    dataset['Companies'] = pd.DataFrame(company_list)
    dataset['Titles_Level'] = pd.DataFrame(title_level_list)
    dataset['Titles'] = pd.DataFrame(title_list)
    print datetime.now()-now
    print 'Data Preparation Completed'
    #trace()

    flag = False

    industry = dataset[dataset['twitter_id']==selected_lead]['Industry'].values[0]
    dataset = pd.DataFrame(dataset[dataset['Industry']==industry])
    industry_cent_list = []
    if industry=='Manufacturing':
        industry_cent_list = Manufacturing_Cent_List
    elif industry=='Media & Entertainment':
        industry_cent_list = Media_Cent_List
    elif industry=='Software':
        industry_cent_list = Software_Cent_List
    elif industry=='Food & Retail':
        industry_cent_list = Food_Cent_List
    elif industry=='Banking & Finance':
        industry_cent_list = Banking_Cent_List
    elif industry=='Services':
        industry_cent_list = Services_Cent_List
    elif industry=='Transportation':
        industry_cent_list = Transportation_Cent_List
    elif industry=='Education':
        industry_cent_list = Education_Cent_List
    elif industry=='Health':
        industry_cent_list = Health_Cent_List
    elif industry=='Sport':
        industry_cent_list = Sport_Cent_List
    else:
        flag = True
        #industry_cent_list = [i for i in range(len(centroid))]
    if flag==False:
        for row, index in dataset.iterrows():
            row['Titles_Level'] = industry_cent_list.index(row['Titles_Level'])
    selected_title_level = dataset[dataset['twitter_id']==selected_lead]['Titles_Level'].values[0]
    dataset = dataset[dataset['Titles_Level'] >= selected_title_level-1]
    dataset = dataset[dataset['Titles_Level'] <= selected_title_level+1]
    selected_company = dataset[dataset['twitter_id']==selected_lead]['Companies'].values[0]
    selected_location = dataset[dataset['twitter_id']==selected_lead]['Location'].values[0]
    selected_interest = dataset[dataset['twitter_id']==selected_lead]['Normalized_Interest'].values[0]
    def calculateMatchScore(record):
        title_level_score = mod_dist(record['Titles_Level'], selected_title_level)
        company_score = fuzz.ratio(record['Companies'], selected_company)
        location_score = [get_loc_score(record['Location'], selected_location)]
        interest_score = [get_interest_score(record['Normalized_Interest'], selected_interest)]
        return pd.Series({'company_score':company_score,
            'industry_score':0,
            'title_level_score':title_level_score,
            'location_score':location_score and max(location_score) or 0,
            'interest_score':interest_score and max(interest_score) or 0,
            'final_score':fin_score(company_score and company_score or 0, 0, title_level_score, location_score and max(location_score) or 0, interest_score and max(interest_score) or 0)})
    tmp = dataset.apply(calculateMatchScore, axis=1)
    dataset[['Company_match', 'Industry_match', 'Title_level_match', 'Final_match', 'Location_Score', 'Interest_Score']] = tmp[['company_score','industry_score','title_level_score', 'final_score', 'location_score', 'interest_score']]

    dataset = dataset[['twitter_id', 'Companies', 'Industry', 'Titles', 'Location', 'interests', 'Company_match', 'Industry_match', 'Title_level_match', 'Location_Score', 'Interest_Score', 'Final_match']]
    print datetime.now() - now
    return dataset


if __name__=='__main__':
    now = datetime.now()
    df = recomendLead('18483485')
    print 'Total Time Taken:', datetime.now() - now
    df.to_csv("test_data_result.csv", index=False)
    print 'Total Time Taken:', datetime.now() - now
    trace()
