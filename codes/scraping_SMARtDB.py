"""
SMARtDB - sequence data

"""

#Import relevant libraries
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import re

f = open("smartdb_sequences.txt",'w')

#Grab the information in the url
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get('http://smartdb.bioinf.med.uni-goettingen.de/cgi-bin/SMARtDB/smar.cgi')
time.sleep(2)

#Find sequence ID and fasta sequence and save in text file
for i in range(2,501):
    try:
        driver.find_element(By.XPATH, '/html/body/font/table/tbody/tr['+str(i)+']/td[1]/font').click()
        time.sleep(2)
        smart_page = driver.find_element(By.XPATH, '/html/body/pre').text
        id_ = re.findall(r"AC  SM0000[0-9]{3}", smart_page)
        id_new = "".join(id_)
        id_new = id_new.replace('AC  ','>')
        
        seq = re.findall(r"\nSQ  [ATGC]+", smart_page)
        #print(id_new)
        f.write(id_new+"\n")
        for j in seq:
            j = j.replace('\nSQ  ','')
            seq_new = "".join(j)      
            #print(seq_new)
            f.write(seq_new)
        f.write("\n\n")
        driver.back()
    except:
        pass












