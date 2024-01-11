import csv



def extract_rows_by_code(input_file_path,codes,output_file, column_to_append):

    '''
    This function iterates through the rows of the large input file and if it matches any codes from the codes list file, it writes the
    entire row in the already created output file. The parameter columns_to_append keeps tha value of the column of the current file 
    we want to append to a list, that we can use for the creation of the next file. If columns_to_append is None, it returns an empty list.
    
    '''
    extracted_rows = []


    with open(input_file_path, 'r', newline='', encoding='utf-8') as csv_file1, open(output_file, 'a', newline='', encoding='utf-8') as csv_file2:
        csv_reader = csv.reader(csv_file1)
        csv_writer = csv.writer(csv_file2, quotechar='"', quoting=csv.QUOTE_ALL)

        for i, row in enumerate(csv_reader):
            if i == 0:
                # Skip the header
                continue

            if  row[0] in codes:
                #csv_writer = csv.writer(csv_file2, quotechar='"', quoting=csv.QUOTE_ALL)
                csv_writer.writerow(row)
                if column_to_append is not None:
                    extracted_rows.append(row[column_to_append-1])


    return extracted_rows


def extract_codes_from_samples(csv_file):

    '''
    Extracts the 1000 sample codes from the sample file and returns a list with those customer codes.

    '''

    codes=[]
    with open(csv_file, 'r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        for row in csv_reader:
            codes.append(row[0])
    
    return codes[1:]


def create_output_file(input_file):

    '''
    This fuction creates the output csv file, where the rows will be later written. It returns the name of the output file.  

    '''
        
    output_file_name=input_file[8:-4]+'_output.csv'
    with open(output_file_name, 'w', newline='', encoding='utf-8') as csv_file1, open(input_file, 'r', newline='', encoding='utf-8') as csv_file2:
    
        csv_reader = csv.reader(csv_file2)
        header_row = next(csv_reader, None)


        if header_row is not None:
            csv_writer = csv.writer(csv_file1)
            csv_writer.writerow(header_row)
    return output_file_name


def create_csv_files(customer_file_path,invoice_file_path,invoice_item_file_path, customer_sample_file_path):
    '''
    This function performs most of the work for the migration. It first creates output files and then populates them with rows
    that match the codes related to the customer sample

    '''

    # Create Customer output file
    customer_file_output_path=create_output_file(customer_file_path)

    #Create Invoice output file
    invoice_file_output_path=create_output_file(invoice_file_path)

    #Create Invoice_item output file
    invoice_item_file_output_path=create_output_file(invoice_item_file_path)

    
    
    #Create a list with 1000 sample customer IDs
    selected_codes=extract_codes_from_samples(customer_sample_file_path)



    #Extract rows from the Customer.csv that match the CUSTOMER_CODE
    customer_codes=extract_rows_by_code(customer_file_path,selected_codes,customer_file_output_path,1)

    #Extract rows from the Invoice.csv that match the CUSTOMER_CODE
    invoice_codes=extract_rows_by_code(invoice_file_path,customer_codes,invoice_file_output_path, 2)

    #Extract rows from the Invoice_Item.csv that match the INVOICE_CODE
    item_codes=extract_rows_by_code(invoice_item_file_path,invoice_codes,invoice_item_file_output_path,None)


    

if __name__=='__main__':

    # Paths of randomly generated input files
    customer_file_path='Testing\Customer.csv'
    invoice_file_path='Testing\Invoice.csv'
    invoice_item_file_path='Testing\Invoice_item.csv'
    customer_sample_file_path='Testing\Customer_sample.csv'
    create_csv_files(customer_file_path,invoice_file_path,invoice_item_file_path, customer_sample_file_path)