
def construct_student_features(df):

    df = df.select('UNITID','CONTROL','INSTNM','ADMIN_TYPE','CIPCODE','CIPDESC','CREDLEV','POSTGRAD','EARN_MDN_HI_1YR','SALARY_GT_40')
    return df




