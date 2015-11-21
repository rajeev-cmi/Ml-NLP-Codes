import wikipedia
from pdb import set_trace as trace
from sklearn.externals import joblib
from datetime import datetime
import pandas as pd

def fetchCompanyDesc(input_file = "input_file.txt"):
    """
    This method reads the input file with company names and pull the information from wiki and return a dataframe.
    """
    company_list = [i.strip() for i in open(input_file, "r").readlines()]
    company_list = list(set(company_list))
    print "Number of companies to pull the information for is :", len(company_list)
    summary_data = []
    for idx, company in enumerate(company_list, start=1):
        #print "Running :", idx
        try:
            summary = wikipedia.summary(company, sentences=3).encode('ascii', 'ignore')
            summary_data.append({"Company":company, "Description":summary})
        except Exception as e:
            print idx, "ERROR :", company, " : ", e
    df = pd.DataFrame(summary_data)
    
    return df
   

if __name__ == "__main__":
    start = datetime.now()
    df = fetchCompanyDesc(input_file = "Iput_File.txt")
    trace()
    df.to_csv("Write_to_File.csv", index=False)
    print "Time Taken for pulling the information is :", datetime.now() - start
