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
def generate_invoice_code():

    row1=select_random_row('Invoice.csv')
    return row1[1]


def generate_item_code():
    item_codes = ["MEIJI", "POCKY", "PUCCHO", "AQUA", "NALA", "INA"]
    return random.choice(item_codes)


def generate_amount():
    return random.uniform(5.0, 800.0)

def generate_quantity():
    return random.randint(0,200)

csv_file_path = 'Invoice_item.csv'

# Number of rows to generate
num_rows = 15000


with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:

    csv_writer = csv.writer(csv_file, quotechar='\"', quoting=csv.QUOTE_ALL)


    csv_writer.writerow(["INVOICE_CODE", "ITEM_CODE", "AMOUNT","QUANTITY"])

    for _ in range(num_rows):
        invoice_code = generate_invoice_code()
        item_code = generate_item_code()
        amount = generate_amount()
        quantity=generate_quantity()

        csv_writer.writerow([invoice_code, item_code, amount, quantity])