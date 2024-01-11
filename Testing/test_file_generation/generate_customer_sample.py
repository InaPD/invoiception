import csv
import random
import string


def select_random_row(csv_file_path):
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)

        if len(rows) > 1:
            selected_row = random.choice(rows[1:])
            return selected_row
        elif len(rows) == 1:
            return rows[0]  
        else:
            return None
def generate_customer_code():
    row1=select_random_row('Testing\Customer.csv')
    return row1[0]


#Number of rows to generate
num_rows = 1000

with open('Testing\Customer_sample.csv', 'w', newline='', encoding='utf-8') as csv_file:

    csv_writer = csv.writer(csv_file, quotechar='"', quoting=csv.QUOTE_ALL)
    csv_writer.writerow(["CUSTOMER_CODE"])

    for _ in range(num_rows):
        customer_code = generate_customer_code()
        csv_writer.writerow([customer_code])