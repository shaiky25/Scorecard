from pyspark.sql import SparkSession
from pyspark.sql.functions import unix_timestamp
from pyspark.sql.functions import trim,when
import numpy as np
import re

# FORMAT = 'CSV'
from pyspark.sql.types import IntegerType, StructField, StructType, DoubleType

FOLDER_PATH = 'data'

COLUMNS = {

    'COLLEGE' : ['UNITID','OPEID6','INSTNM','CONTROL','ST_FIPS','NPT4_PUB','NPT41_PUB','NPT42_PUB','NPT43_PUB','NPT44_PUB','NPT45_PUB',
                 'NPT4_PRIV','NPT41_PRIV','NPT42_PRIV','NPT43_PRIV','NPT44_PRIV','NPT45_PRIV',
                 'ADM_RATE','SATVRMID','SATMTMID','SATWRMID','ACTCMMID','ACTENMID','ACTMTMID','ACTWRMID',
                 'C150_4','C150_L4'],
    'FIELD_OF_STUDY':['UNITID','OPEID6','INSTNM','CONTROL','CIPCODE','CIPDESC','CREDLEV','EARN_COUNT_NWNE_HI_1YR',
                      'EARN_CNTOVER150_HI_1YR','EARN_COUNT_WNE_HI_1YR','EARN_COUNT_NWNE_HI_2YR',
                      'EARN_CNTOVER150_HI_2YR','EARN_COUNT_WNE_HI_2YR','EARN_MDN_HI_1YR','EARN_MDN_HI_2YR']

    }

# College details : 'UNITID','OPEID6','INSTNM','CONTROL','STABBR'
# Costs :'NPT41_PUB','NPT42_PUB','NPT43_PUB','NPT44_PUB','NPT45_PUB','NPT41_PRIV','NPT42_PRIV','NPT43_PRIV','NPT44_PRIV','NPT45_PRIV',
# admission rates & scores : 'ADM_RATE','SATVRMID','SATMTMID','SATWRMID','ACTCMMID','ACTENMID','ACTMTMID','ACTWRMID'
# completion rates :'C150_4','C150_L4'
# field of study or credential level: 'CIPCODE','CIPDESC','CREDLEV'
# earnings :'EARN_COUNT_NWNE_HI_1YR', 'EARN_CNTOVER150_HI_1YR','EARN_COUNT_WNE_HI_1YR','EARN_MDN_HI_1YR',
#           'EARN_COUNT_NWNE_HI_2YR','EARN_CNTOVER150_HI_2YR','EARN_COUNT_WNE_HI_2YR','EARN_MDN_HI_2YR'

def _read_data(table):
    #spark = SparkSession.builder.appName('CollegeScoreCard').getOrCreate()
    spark = SparkSession.builder \
        .master('local[*]') \
        .config("spark.driver.memory", "15g") \
        .appName('CollegeScoreCard') \
        .getOrCreate()
    if table == 'COLLEGE' :
        df= spark.read.csv(f'{FOLDER_PATH}/MERGED2017_18_PP.csv',header=True, inferSchema=True)
        # df= spark.read.csv(f'{FOLDER_PATH}',header=True, inferSchema=True)
        df = df.select(COLUMNS[table])
        print("Before Cleaning ",df.count())
        return df
    elif table == 'FIELD_OF_STUDY' :
        df= spark.read.csv(f'{FOLDER_PATH}/FOS',  header=True, inferSchema=True)
        df = df.select(COLUMNS[table])
        print("Before Cleaning ",df.count())

        return df


def clean_data(df,table):
     if table == 'COLLEGE' :
         df = df.na.drop(subset=['NPT4_PUB','NPT4_PRIV','NPT41_PUB','NPT42_PUB','NPT43_PUB','NPT44_PUB','NPT45_PUB',
                 'NPT41_PRIV','NPT42_PRIV','NPT43_PRIV','NPT44_PRIV','NPT45_PRIV',
                 'ADM_RATE','SATVRMID','SATMTMID','SATWRMID','ACTCMMID','ACTENMID','ACTMTMID','ACTWRMID',
                 'C150_4','C150_L4'])
         for c_name in df.columns:
            df = df.withColumn(c_name, trim(df[c_name]))
         df =df.where("NPT4_PUB!='NULL' or NPT4_PRIV!='NULL' or NPT41_PUB!='NULL' or NPT42_PUB!='NULL' or NPT43_PUB!='NULL' or NPT44_PUB!='NULL' or NPT45_PUB!='NULL'"
                     "or NPT41_PRIV!='NULL' or NPT42_PRIV!='NULL' or NPT43_PRIV!='NULL' or NPT44_PRIV!='NULL'  or NPT45_PRIV!='NULL'"
                     "or ADM_RATE!='NULL' or SATVRMID !='NULL' or SATMTMID !='NULL' or SATWRMID!='NULL' or ACTCMMID !='NULL' "
                     "or ACTENMID !='NULL' or ACTMTMID !='NULL' or ACTWRMID !='NULL' or C150_4 !='NULL' or C150_L4 !='NULL'")
         df= df.withColumn("NPT41_PUB", when(df["NPT41_PUB"] == 'NULL', '0')
                                       .otherwise(df["NPT41_PUB"]))
         df= df.withColumn("NPT41_PRIV", when(df["NPT41_PRIV"] == 'NULL', '0')
                                           .otherwise(df["NPT41_PRIV"]))
         df= df.withColumn("ADM_RATE", when(df["ADM_RATE"] == 'NULL', '0')
                                           .otherwise(df["ADM_RATE"]))
         df= df.withColumn("C150_4", when(df["C150_4"] == 'NULL', '0')
                                           .otherwise(df["C150_4"]))
         df= df.withColumn("C150_L4", when(df["C150_L4"] == 'NULL', '0')
                                           .otherwise(df["C150_L4"]))

         return df
     elif table == 'FIELD_OF_STUDY' :
         for c_name in df.columns:
            df = df.withColumn(c_name, trim(df[c_name]))
         #     Removing all nulls and Any rows where EARNINGS for 1 year is Privacy Supressed
         df = df.na.drop(subset=['EARN_COUNT_NWNE_HI_1YR',
                      'EARN_CNTOVER150_HI_1YR','EARN_COUNT_WNE_HI_1YR','EARN_MDN_HI_1YR','EARN_COUNT_NWNE_HI_2YR',
                      'EARN_CNTOVER150_HI_2YR','EARN_COUNT_WNE_HI_2YR','EARN_MDN_HI_2YR'])
         df =df.where("(EARN_COUNT_NWNE_HI_1YR!='PrivacySuppressed' or EARN_CNTOVER150_HI_1YR!='PrivacySuppressed' or EARN_COUNT_WNE_HI_1YR!='PrivacySuppressed' "
                     "or EARN_COUNT_NWNE_HI_2YR!='PrivacySuppressed' or EARN_CNTOVER150_HI_2YR!='PrivacySuppressed' "
                     "or EARN_COUNT_WNE_HI_2YR!='PrivacySuppressed' or EARN_MDN_HI_2YR!='PrivacySuppressed')"
                     "and EARN_MDN_HI_1YR!='PrivacySuppressed' ")
          #     Removing all rows which don't have UNITID
         df =df.where("EARN_MDN_HI_1YR!='NULL' and UNITID!='NULL' and CONTROL != 'NULL'")
         # df= df.filter(df["CONTROL"] != 'NULL')
         df= df.withColumn("ADMIN_TYPE", when(df["CONTROL"] == 'Public', 1)
                      .otherwise(0))
         df =df.withColumn("CREDLEV",df["CREDLEV"].cast(IntegerType()))
         df =df.withColumn("CIPCODE",df["CIPCODE"].cast(IntegerType()))
         df =df.withColumn("UNITID",df["UNITID"].cast(IntegerType()))
         df =df.withColumn("OPEID6",df["OPEID6"].cast(IntegerType()))
         df =df.withColumn("EARN_MDN_HI_1YR",df["EARN_MDN_HI_1YR"].cast(IntegerType()))
         df =df.withColumn("POSTGRAD", when(df["CREDLEV"] <=3, 0)
                      .otherwise(1))
         df =df.withColumn("SALARY_GT_40", when(df["EARN_MDN_HI_1YR"] <=40000, 0)
                      .otherwise(1))
         return df

def build_dataframe(table):
    df = _read_data(table)
    df = clean_data(df,table)
    return df

def groupCIPdesc(df):
 df['STEM'] = 0
 df = df.replace(np.nan, 0, regex=True)
 df.loc[(df['CIPDESC'].str.contains(pat='SCIENCE',flags=re.IGNORECASE)) \
        |(df['CIPDESC'].str.contains(pat='COMPUTER',flags=re.IGNORECASE)) \
        |(df['CIPDESC'].str.contains(pat='HEALTH',flags=re.IGNORECASE)) \
        |(df['CIPDESC'].str.contains(pat='THERAP',flags=re.IGNORECASE)) \
        |(df['CIPDESC'].str.contains(pat='TECH',flags=re.IGNORECASE)) \
        |(df['CIPDESC'].str.contains(pat='MEDICINE',flags=re.IGNORECASE)) \
        |(df['CIPDESC'].str.contains(pat='MEDICAL',flags=re.IGNORECASE)) \
        |(df['CIPDESC'].str.contains(pat='DENTAL',flags=re.IGNORECASE)) \
        |(df['CIPDESC'].str.contains(pat='MATH',flags=re.IGNORECASE)) \
        |(df['CIPDESC'].str.contains(pat='ENGINE',flags=re.IGNORECASE)), "STEM"] = 1

 # df = df.drop('CIPDESC', axis=1)
 return df
