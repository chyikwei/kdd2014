import pandas as pd
import scipy.io as sio
import numpy as np
import scipy.sparse as sp


FULL_PROJECTS_FILE_NAME = 'raw_data/projects.csv'
PROJECT_FILE_NAME = 'projects_train_id_only.csv'
OUTCOME_FILE_NAME = 'train_is_excited'

def main():
    projects = pd.read_csv(PROJECT_FILE_NAME)
    projects = projects.sort('projectid')
    outcomes = pd.read_csv(OUTCOME_FILE_NAME)
    #outcomes = outcomes.sort('projectid')
    projects['is_exciting'] = outcomes['is_exciting']

    #proj_oc = pd.merge(projects, outcomes, on='projectid')

    # get history on teacher_id
    grouped = projects.groupby('schoolid')

    ids = []
    project_cnts = []
    exciting_project_cnts = []
    for name, group in grouped:
        proj_cnt = group.shape[0]
        exciting_proj_cnt = sum(group['is_exciting'] == 't')

        ids.append(name)
        project_cnts.append(proj_cnt)
        exciting_project_cnts.append(exciting_proj_cnt)


    # save file
    data = {
        'schoolid': ids,
        'school_proj_cnt': project_cnts,
        'school_exciting_proj_cnt': exciting_project_cnts,
    }

    df = pd.DataFrame(data)
    df.to_csv('schoolid_history.csv', index=False)

def add_history_to_projects():
    FULL_PROJECTS_FILE_NAME = 'raw_data/projects.csv'
    TEACHER_FILE_NAME = 'teacher_history.csv'
    SCHOOL_FILE_NAME = 'schoolid_history.csv'
    projects = pd.read_csv(FULL_PROJECTS_FILE_NAME)
    teacher_history = pd.read_csv(TEACHER_FILE_NAME)
    schoolid_history = pd.read_csv(SCHOOL_FILE_NAME)
    projects_teacher = pd.merge(projects, teacher_history, on='teacher_acctid', how='left')
    projects_teacher_school = pd.merge(projects_teacher, schoolid_history, on='schoolid', how='left')

def add_fields():
    def get_teacher_exciting_before(row):
        if pd.isnull(row['teacher_exciting_cnt']):
            return
        else:
            if row['teacher_exciting_cnt'] > 0:
                return 1.0
            else:
                return 0.0

    def get_teacher_exciting_pct(row):
        if pd.isnull(row['teacher_exciting_cnt']):
            return
        else:
            if row['teacher_exciting_cnt'] > 0:
                return float(row['teacher_exciting_cnt']) / row['teacher_proj_cnt']
            else:
                return 0.0

    def get_school_exciting_before(row):
        if pd.isnull(row['school_exciting_proj_cnt']):
            return
        else:
            if row['school_exciting_proj_cnt'] > 0:
                return 1.0
            else:
                return 0.0

    def get_school_exciting_pct(row):
        if pd.isnull(row['school_exciting_proj_cnt']):
            return
        else:
            if row['school_exciting_proj_cnt'] > 0:
                return float(row['school_exciting_proj_cnt']) / row['school_proj_cnt']
            else:
                return 0.0


    projects_teacher_school['teacher_have_exciting'] = projects_teacher_school.apply(get_teacher_exciting_before, axis=1)
    projects_teacher_school['teacher_exciting_pct'] = projects_teacher_school.apply(get_teacher_exciting_pct, axis=1)
    projects_teacher_school['school_have_exciting'] = projects_teacher_school.apply(get_school_exciting_before, axis=1)
    projects_teacher_school['school_exciting_pct'] = projects_teacher_school.apply(get_school_exciting_pct, axis=1)

def to_matrix():
    train_2013 = projects_teacher_school[((projects_teacher_school['date_posted'] < '2014-01-01') & (projects_teacher_school['date_posted'] >= '2013-01-01'))]
    train_2013 = train_2013.sort('projectid')

    fields = [
        #'teacher_exciting_cnt',
        'teacher_proj_cnt',
        #'teacher_have_exciting',
        #'teacher_exciting_pct',
        #'school_exciting_proj_cnt',
        'school_proj_cnt',
        #'school_have_exciting',
        #'school_exciting_pct',
    ]

    train_data = []
    for field in fields:
        train_data.append(train_2013[field])

    train_mtx = np.vstack(train_data).T
    train_mtx = sp.csr_matrix(train_mtx)
    sio.mmwrite('train_2013_t_school_cnt_mtx', train_mtx)

    test_data = projects_teacher_school[projects_teacher_school['date_posted'] >= '2014-01-01']
    test_data = test_data.sort('projectid')

    test_data['teacher_exciting_cnt'] = test_data['teacher_exciting_cnt'].fillna(0.0)
    test_data['teacher_proj_cnt'] = test_data['teacher_exciting_cnt'].fillna(1.0)
    test_data['teacher_have_exciting'] = test_data['teacher_have_exciting'].fillna(0.0)
    test_data['teacher_exciting_pct'] = test_data['teacher_exciting_pct'].fillna(0.1553)

    test_data['school_exciting_proj_cnt'] = test_data['school_exciting_proj_cnt'].fillna(0.0)
    test_data['school_proj_cnt'] = test_data['school_proj_cnt'].fillna(1.0)
    test_data['school_have_exciting'] = test_data['school_have_exciting'].fillna(0.0)
    test_data['school_exciting_pct'] = test_data['school_exciting_pct'].fillna(0.0657)

    test_data_list = []
    for field in fields:
        test_data_list.append(test_data[field])

    test_mtx = np.vstack(test_data_list).T
    test_mtx = sp.csr_matrix(test_mtx)
    sio.mmwrite('test_t_school_cnt_mtx', test_mtx)
