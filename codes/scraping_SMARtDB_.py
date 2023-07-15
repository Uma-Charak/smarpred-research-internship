#Import relevant libraries
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

# Open the file to write sequence data
with open("smartdb_sequences.txt", "w") as f:
    # Set up the web driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get('http://smartdb.bioinf.med.uni-goettingen.de/cgi-bin/SMARtDB/smar.cgi')
    time.sleep(2)

    # Find sequence ID and fasta sequence and save in text file
    for i in range(2, 501):
        try:
            # Click on the sequence link
            sequence_link = driver.find_element(By.XPATH, f'/html/body/font/table/tbody/tr[{i}]/td[1]/font')
            sequence_link.click()
            time.sleep(2)

            # Get the sequence information
            smart_page = driver.find_element(By.XPATH, '/html/body/pre').text
            sequence_id = re.findall(r"AC  SM0000[0-9]{3}", smart_page)
            sequence_id = "> " + "".join(sequence_id).replace('AC  ', '') + "\n"

            sequence = re.findall(r"\nSQ  [ATGC]+", smart_page)
            sequence = "".join([s.replace('\nSQ  ', '') for s in sequence]) + "\n\n"

            # Write sequence information to file
            f.write(sequence_id)
            f.write(sequence)

            # Go back to the previous page
            driver.back()
        except:
            pass

    # Close the web driver
    driver.quit()
