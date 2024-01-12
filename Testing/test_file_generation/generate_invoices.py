import csv
import random
import string
import datetime


def select_random_row(csv_file_path):
    with open(csv_file_path, 'r', newline='', encoding='ascii') as csv_file:
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

def generate_invoice_code():
    return 'IN' + ''.join(random.choices(string.digits, k=7))

def generate_amount():
    return random.uniform(5.0, 800.0)

def generate_date():
    return str(datetime.datetime.now())


csv_file_path = r'Testing\Invoice.csv'

# Number of rows to generate
num_rows = 2500


with open(csv_file_path, 'w', newline='', encoding='ascii') as csv_file:

    csv_writer = csv.writer(csv_file, quotechar='\"', quoting=csv.QUOTE_ALL)

    csv_writer.writerow(["CUSTOMER_CODE", "INVOICE_CODE", "AMOUNT","DATE"])


    for _ in range(num_rows):
        customer_code = generate_customer_code()
        invoice_code = generate_invoice_code()
        amount = generate_amount()
        date=generate_date()

        csv_writer.writerow([customer_code, invoice_code, amount, date])