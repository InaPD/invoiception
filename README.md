## DreamCandies File Tool

This project is used as a POC migration tool for .csv files by DreamCandies. The idea is to select a random sample of 1000 customers from a large dataset and use their customer codes to extract data related to those customers. The data related to the customers should be extracted from 3 .csv files that are received on input: Customer.csv, Invoice.csv and Invoice_item.csv. Since those example files were not given I generated random test datasets for those files according to the description in the assignment text. The test datasets can be found in the Testing folder.

The requirements of the assignment are:

- From the **Customer.csv** file we should extract the rows where the customer code matches the customer codes from the sample file and write them into a separate output file. 
- From the **Invoice.csv** file we should extract the row that match the customer codes from the sample file and record them in an output file.
- And from the **Invoice_item.csv** we should select the rows that match the invoice code field with the invoices previously selected for the invoice output file. 

In order to perform this functionality, I first create the empty output files. Then I iterate through all 3 .csv files, match the designated column to the codes as needed and write down the rows in the output files. 

