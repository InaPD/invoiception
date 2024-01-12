import csv
import random


csv_file_path = 'Testing\Customer.csv'

num_random_rows = 1000

with open(csv_file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  
    all_rows = list(reader)


if len(all_rows) < num_random_rows:
    print("The CSV file doesn't have enough rows.")
else:

    random_rows = random.sample(all_rows, num_random_rows)


    with open('Testing\Customer_sample_file.csv', 'w', newline='', encoding='ascii') as csvfile:
        writer = csv.writer(csvfile,  quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow([header[0]])  
        for row in random_rows:
            writer.writerow([row[0]])