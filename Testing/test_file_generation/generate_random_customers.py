import csv
import random
import string


def generate_customer_code():
    return 'CUST' + ''.join(random.choices(string.digits, k=10))


def generate_first_name():
    first_names = ["Maria", "George", "John", "Alice", "Bob", "Eva", "David", "Sophia"]
    return random.choice(first_names)


def generate_last_name():
    last_names = ['Alba', 'Lucas', 'Smith', 'Johnson', 'Garcia', 'Lee', 'Wang', 'Kim']
    return random.choice(last_names)


csv_file_path = r'Testing\Customer.csv'


num_rows = 1500


with open(csv_file_path, 'w', newline='', encoding='ascii') as csv_file:

    csv_writer = csv.writer(csv_file, quotechar='"', quoting=csv.QUOTE_ALL)


    csv_writer.writerow(["CUSTOMER_CODE", "FIRSTNAME", "LASTNAME"])

    for _ in range(num_rows):
        customer_code = generate_customer_code()
        first_name = generate_first_name()
        last_name = generate_last_name()

        csv_writer.writerow([customer_code, first_name, last_name])