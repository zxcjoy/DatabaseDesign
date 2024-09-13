import pandas as pd
import os
import re
import shutil
from datetime import datetime

def update_metadata(table_name, locations, header, current_row, total_file_size, create_time, modified_time):
    df_metadata = pd.read_csv('tables_metadata.csv')
    
    new_row = pd.DataFrame({
        'table_name': [table_name],
        'locations': [locations],
        'header': [','.join(header)],
        'current_row': [current_row],
        'total_file_size': [total_file_size],
        'create_time': [create_time],
        'modified_time': [modified_time]
    })

    if table_name in df_metadata['table_name'].values:
        # Update existing entry
        df_metadata.loc[df_metadata['table_name'] == table_name, ['locations', 'current_row', 'total_file_size', 'modified_time']] = [locations, current_row, total_file_size, modified_time]
    else:
        # Concatenate new row
        df_metadata = pd.concat([df_metadata, new_row], ignore_index=True)
    
    df_metadata.to_csv('tables_metadata.csv', index=False)

def convert_data_types(values):
    def convert(value):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    if isinstance(values, list):
        return [convert(value) for value in values]
    else:
        return convert(values)

def check_and_handle_existing_table(table_name):
    df_metadata = pd.read_csv('tables_metadata.csv')
    if table_name in df_metadata['table_name'].values:
        user_input = input(f"Table '{table_name}' already exists. Type 'delete' to remove it, or 'skip' to cancel operation: ").strip().lower()
        if user_input == 'delete':
            file_directory = os.path.join(os.getcwd(), 'files', table_name)
            if os.path.exists(file_directory):
                shutil.rmtree(file_directory)
            df_metadata = df_metadata[df_metadata['table_name'] != table_name]
            df_metadata.to_csv('tables_metadata.csv', index=False)
            print(f"Table '{table_name}' and its data have been deleted.")
            return True
        elif user_input == 'skip':
            print("Operation skipped.")
            return False
    return True

def split_and_save_csv(file_path, table_name, max_rows=9999):
    if not check_and_handle_existing_table(table_name):
        return

    base_name = os.path.basename(file_path)
    file_directory = os.path.join(os.getcwd(), 'files/', table_name + '/')
    os.makedirs(file_directory, exist_ok=True)

    # total_file_size = 0
    create_time = datetime.now()
    total_chunks = sum(1 for _ in pd.read_csv(file_path, chunksize=max_rows, low_memory=False))  # Count total number of chunks
    print('Begin reading file into database...')
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize = max_rows, low_memory=False)):
        
        new_file_name = f"{file_directory}{table_name}_{i}.csv"
        chunk.to_csv(new_file_name, index=False)
        
        if i == total_chunks - 1:
            total_file_size = sum([os.path.getsize(os.path.join(file_directory, f)) for f in os.listdir(file_directory) if f.endswith('.csv')])
            update_metadata(table_name, i, chunk.columns.tolist(), len(chunk)+1, total_file_size, create_time, datetime.now())

        # total_file_size += os.path.getsize(new_file_name)
        # update_metadata(table_name, i, chunk.columns.tolist(), len(chunk), total_file_size, create_time, datetime.now())

    print(f"Finished splitting {base_name} into smaller files in {file_directory}")

def import_from_file(command):            
    pattern = r'IMPORT FROM (\w+\.csv) AS (\w+)'
    match = re.match(pattern, command)
    if match:
        target_file_name, table_name = match.groups()
        split_and_save_csv(target_file_name, table_name)
    else:
        print("Invalid command format.")

def print_metadata():
    metadata_file = 'tables_metadata.csv'
    if os.path.exists(metadata_file):
        df_metadata = pd.read_csv(metadata_file)
        print(df_metadata)
    else:
        print("Metadata file does not exist.")

def delete_metadata_entry(table_name):
    # Delete the metadata entry
    df_metadata = pd.read_csv('tables_metadata.csv')
    df_metadata = df_metadata[df_metadata['table_name'] != table_name]
    df_metadata.to_csv('tables_metadata.csv', index=False)
    print(f"Metadata entry for table '{table_name}' has been deleted.")

def delete_table_files_and_metadata(table_name, file_directory):
    # Delete all files in the directory
    shutil.rmtree(file_directory)
    print(f"All files for table '{table_name}' have been deleted.")
    delete_metadata_entry(table_name)

def check_all_tables_integrity():
    metadata_file = 'tables_metadata.csv'
    if not os.path.exists(metadata_file):
        print("Metadata file does not exist.")
        return

    df_metadata = pd.read_csv(metadata_file)
    for _, row in df_metadata.iterrows():
        table_name = row['table_name']
        file_directory = os.path.join(os.getcwd(), 'files', table_name)

        # directory for table_name does not exist
        if not os.path.exists(file_directory):
            print(f"Directory for table '{table_name}' does not exist.")
            user_input = input("Do you want to delete the metadata entry for this table? (yes/no): ").strip().lower()
            if user_input == 'yes':
                delete_metadata_entry(table_name)
            else:
                print("No action taken for table '{table_name}'.")
            continue
        
        expected_files = row['locations'] + 1
        expected_size = row['total_file_size']
        actual_files = sum([1 for _ in os.listdir(file_directory) if _.startswith(table_name) and _.endswith('.csv')])
        actual_size = sum([os.path.getsize(os.path.join(file_directory, f)) for f in os.listdir(file_directory) if f.endswith('.csv')])

        if actual_files != expected_files or actual_size != expected_size:
            print(f"Integrity check failed for table '{table_name}'.")
            user_input = input("Do you want to delete the damaged table and its metadata? (yes/no): ").strip().lower()
            if user_input == 'yes':
                delete_table_files_and_metadata(table_name, file_directory)
            else:
                print("No action taken for table '{table_name}'.")
        else:
            print(f"Table '{table_name}' is complete and consistent with the metadata.")

def calculate_total_size(directory):
    return sum(os.path.getsize(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.csv'))

def insert_query_handle(command):

    pattern = r'ADD TO TABLE (\w+) WITH VALUES \[(.+)\]'
    match = re.match(pattern, command)
    if match:
        table_name, values_str = match.groups()
        values = convert_data_types([value.strip() for value in values_str.split(',')])
        insert_data_to_table(table_name, values)
    else:
        print('Query failed. Please check the input command.')

def insert_data_to_table(file_path, data):
    df_metadata = pd.read_csv('tables_metadata.csv')
    if file_path not in df_metadata['table_name'].values:
        print(f"Table '{file_path}' does not exist.")
        return

    metadata_row = df_metadata[df_metadata['table_name'] == file_path].iloc[0]
    locations = int(metadata_row['locations'])
    current_row = int(metadata_row['current_row'])
    header = metadata_row['header'].split(',')
    file_directory = os.path.join(os.getcwd(), 'files', file_path)

    # Check if a new file is needed
    if current_row >= 10000:
        locations += 1
        current_row = 1 # include the header
        new_file_name = f"{file_directory}/{file_path}_{locations}.csv"
        pd.DataFrame(columns=header).to_csv(new_file_name, index=False)  # Create a new file with header
    else:
        new_file_name = f"{file_directory}/{file_path}_{locations}.csv"

    # Append data to the file
    with open(new_file_name, 'a') as f:
        f.write(','.join(map(str, data)) + '\n')  # Ensure data is converted to string

    # Update metadata
    total_file_size = sum([os.path.getsize(os.path.join(file_directory, f)) for f in os.listdir(file_directory) if f.startswith(file_path) and f.endswith('.csv')])
    update_metadata(file_path, locations, header, current_row + 1, total_file_size, metadata_row['create_time'], datetime.now())
    print("Data inserted.")

def delete_query_handle(command):
    pattern = r"REMOVE FROM TABLE (\w+) THAT (\w+) (<|>|=|!=) '([\w\d]+)'"
    # pattern = r"REMOVE FROM TABLE (\w+) THAT (\w+) (<|>|=|!=) '([^']*)'"
    match = re.match(pattern, command)
    if match:
        table_name, column, operator, value = match.groups()
        value = convert_data_types(value)
        delete_data_from_table(table_name, column, operator, value)
    else:
        print('Query failed. Please check the input command.')

def delete_data_from_table(table_name, column, operator, value):
    df_metadata = pd.read_csv('tables_metadata.csv')
    if table_name not in df_metadata['table_name'].values:
        print(f"Table '{table_name}' does not exist.")
        return

    metadata_row = df_metadata[df_metadata['table_name'] == table_name].iloc[0]
    locations = int(metadata_row['locations']) + 1
    file_directory = os.path.join(os.getcwd(), 'files', table_name)

    for i in range(locations):
        file_path = f"{file_directory}/{table_name}_{i}.csv"
        df = pd.read_csv(file_path)

        # Apply the condition
        if operator == '<':
            df = df[~(df[column] < value)]
        elif operator == '>':
            df = df[~(df[column] > value)]
        elif operator == '=':
            df = df[~(df[column] == value)]
        elif operator == '!=':
            df = df[~(df[column] != value)]  # Use the converted value directly

        # Rewrite the file
        df.to_csv(file_path, index=False)

    # Update metadata
    update_metadata(table_name, locations - 1, list(df.columns), df.shape[0], calculate_total_size(file_directory), metadata_row['create_time'], datetime.now())
    print(f"Data removed from table '{table_name}'.")

def print_tables_name():
    metadata_file = 'tables_metadata.csv'
    try:
        df_metadata = pd.read_csv(metadata_file)
        if not df_metadata.empty:
            for table_name in df_metadata['table_name']:
                print(table_name)
        else:
            print("No tables found in the database.")
    except FileNotFoundError:
        print("Metadata file not found.")

def show_table_rows(command):
    pattern = r'SHOW (\w+) LIMIT (\d+)'
    match = re.match(pattern, command)

    if match:
        table_name, row_limit_str = match.groups()
        row_limit = int(row_limit_str)

        file_directory = os.path.join(os.getcwd(), 'files', table_name)
        file_path = os.path.join(file_directory, f"{table_name}_0.csv")

        if not os.path.exists(file_path):
            print(f"No data file found for table '{table_name}'.")
            return

        df = pd.read_csv(file_path)
        rows_to_show = min(row_limit, len(df))
        print(df.head(rows_to_show).to_string(index=False))
    else:
        print("Invalid command format.")

def system_initiate():
    # Global metadata file
    metadata_file = 'tables_metadata.csv'
    # Initialize metadata file if it doesn't exist
    if not os.path.exists(metadata_file):
        metadata_file_header = ['table_name', 'locations', 'header', 'current_row',
                           'total_file_size', 'create_time', 'modified_time']
        pd.DataFrame(columns=metadata_file_header).to_csv(metadata_file, index=False)
    welcome_notes = '''Welcome to OurDB!
Created for DSCI 551 project.
Author: Ziyu Chen'''
    print(welcome_notes)
    print('System initiating...')
    print('Checking data integrity...')
    check_all_tables_integrity()
    print('System initiating completed.')
    print('Type HELP to see the reference.')

def print_help():
    reference_message = '''
# **User Handbook for OurDB Custom Database System**

## **Introduction**

OurDB is a versatile and user-friendly database management system designed to handle various data operations. This handbook serves as a guide to effectively utilize the features of OurDB.

## **Getting Started**

- Start the system by running the main script.
- Type your commands at the **`OurDB >`** prompt.
- To exit the system, type **`EXIT`** or **`E`**.

## **Commands**

### **General Commands**

- **Help**: Type **`HELP`** to display help text with command references.
- **Exit**: Type **`EXIT`** or **`E`** to exit the database system.
- **Clear Database**: Type **`CLEAR DATABASE`** to delete all data and reset the system. Requires confirmation.
- **Show Tables**: Type **`SHOW TABLES`** to view all tables and their metadata.
- **Show Table Names**: Type **`SHOW TABLES NAME`** to list the names of all available tables in the database.

### **Table Management**

- **Create Table**: To create a new table, use **`CREATE TABLE <table_name> [<column1>, <column2>, ...] WITH VALUES [<value1>, <value2>, ...]`**. Example: **`CREATE TABLE my_table [name, revenue, employee] WITH VALUES [CompanyA, 1000000, 50]`**.
- **Import Table from File**: Use **`IMPORT FROM <filename.csv> AS <table_name>`** to create a new table from a CSV file.
- **Delete Table**: To delete a specific table, use **`DELETE TABLE <table_name>`**. Example: **`DELETE TABLE car`**.

### **Data Manipulation**

- **Insert Data**: To add data to a table, use **`ADD TO TABLE <table_name> WITH VALUES [<value1>, <value2>, ...]`**.
- **Delete Data**: To remove data from a table based on a condition, use **`REMOVE FROM TABLE <table_name> THAT <column> <operator> <value>`**.
- **Find Data**: Use **`FIND`** commands to query data. Examples include **`FIND ALL FROM <table_name> WHERE <condition>`**, **`FIND <aggregation>(<column>) FROM <table_name> FOR EACH <group_column> WITH ORDER ASC/DESC`**
    
    Example:
    
    FIND ALL FROM <table_name> (optional WHERE)
    
    FIND SUM(column_name) FROM <table_name> FOR EACH <group_column_name> (optional WITH ORDER ASC)
    
- **Aggreagation function**: Currently support SUM, MIN, MAX, COUNT
- **Sorting method**: Currently support ascending and descending, only used after a FIND query
- **Update Data**: To update data in a table, use **`UPDATE TO TABLE <table_name> LET <column> = <new_value> [WHERE <condition>]`**.

### **Join Operation**

- **Join Tables**: To join two tables, use **`JOIN <table1> WITH <table2> AS <new_table_name> ON THAT <table1.column1> = <table2.column2>`**. The result is saved as a new table.

### **Advanced Features**

- **Show Specific Rows**: To display a specific number of rows from a table, use **`SHOW <table_name> LIMIT <number>`**.
- **Integrity Check**: Automatically checks the integrity of tables against their metadata and offers options to handle inconsistencies.
'''
    print(reference_message)

def apply_operation(current_value, operation, column_name):
    local_dict = {column_name: current_value}
    try:
        return eval(operation, {"__builtins__": {}}, local_dict)
    except Exception as e:
        print(f"Error applying operation: {e}")
        return current_value

def update_table(command):
    # Check if command contains a WHERE clause
    has_where = ' WHERE ' in command.upper()

    if has_where:
        pattern = r"UPDATE TO TABLE (\w+) LET (\w+) = ([\w\s\*\/\+\-\.]+) WHERE (\w+) ([!=><]+) '([^']+)'"
    else:
        pattern = r"UPDATE TO TABLE (\w+) LET (\w+) = ([\w\s\*\/\+\-\.]+)"

    match = re.match(pattern, command)
    
    if match:
        # print(match.groups())
        table_name, column_to_update, operation = match.groups()[:3]
        cond_column, operator, cond_value = match.groups()[3:] if has_where else (None, None, None)

        if cond_column and cond_value:
            cond_value = convert_data_types(cond_value)

        df_metadata = pd.read_csv('tables_metadata.csv')
        if table_name not in df_metadata['table_name'].values:
            print(f"Table '{table_name}' does not exist.")
            return

        metadata_row = df_metadata[df_metadata['table_name'] == table_name].iloc[0]
        locations = int(metadata_row['locations']) + 1
        file_directory = os.path.join(os.getcwd(), 'files', table_name)

        for i in range(locations):
            file_path = f"{file_directory}/{table_name}_{i}.csv"
            df = pd.read_csv(file_path)

            operation_formatted = operation.replace(column_to_update, f"{column_to_update}")
            if cond_column:
                condition = df[cond_column].astype(str) == str(cond_value)
                df.loc[condition, column_to_update] = df.loc[condition, column_to_update].apply(lambda x: apply_operation(x, operation_formatted, column_to_update))
            else:
                df[column_to_update] = df[column_to_update].apply(lambda x: apply_operation(x, operation_formatted, column_to_update))

            df.to_csv(file_path, index=False)

        print(f"Table '{table_name}' has been updated.")
    else:
        print("Invalid command format.")

def create_table(command):
    # Regular expression pattern for parsing the command
    pattern = r"CREATE TABLE (\w+) \[([^\]]+)\] WITH VALUES \[([^\]]+)\]"
    match = re.match(pattern, command)

    if match:
        table_name, columns, values = match.groups()
        columns = columns.split(',')
        values = values.split(',')

        # Convert values to appropriate data types
        values = convert_data_types(values)

        # Create a DataFrame with initial values
        df = pd.DataFrame([values], columns=columns)

        # Create a directory for the new table and save the DataFrame
        file_directory = os.path.join(os.getcwd(), 'files', table_name)
        os.makedirs(file_directory, exist_ok=True)
        file_path = os.path.join(file_directory, f"{table_name}_0.csv")
        df.to_csv(file_path, index=False)

        # Calculate file size
        total_file_size = os.path.getsize(file_path)

        # Update metadata
        update_metadata(table_name, 0, columns, 2, total_file_size, datetime.now(), datetime.now())

        print(f"Table '{table_name}' created with initial values.")
    else:
        print("Invalid CREATE TABLE command.")

def execute_select_query(command):
    # Regular expressions for different types of queries
    agg_segment = re.search(r"FIND (\w+)\((.*?)\) FROM (\w+)", command)  # Aggregation function
    column_segment = re.search(r"FIND (\w+) FROM (\w+)", command)  # Specific column
    all_segment = re.search(r"FIND ALL FROM (\w+)", command)  # All columns
    group_segment = re.search(r"FOR EACH (\w+)", command)
    order_segment = re.search(r"WITH ORDER (ASC|DESC)", command)
    where_segment = re.search(r"WHERE (\w+) ([!=><]+) '([^']+)'", command)  # Enhanced WHERE clause
    
    # if agg_segment:
    #     print(agg_segment.groups())
    # else:
    #     print('No agg segment')
    # if column_segment:
    #     print(column_segment.groups())
    # else:
    #     print('No column segment')
    # if all_segment:
    #     print(all_segment.groups())
    # else:
    #     print('No ALL segment')
    # if group_segment:
    #     print(group_segment.groups())
    # else:
    #     print('No group segment')
    # if order_segment:
    #     print(order_segment.groups())
    # else:
    #     print('No order segment')
    # if where_segment:
    #     print(where_segment.groups())
    # else:
    #     print('No where segment')

    # Determine the type of query
    if agg_segment:
        agg_func, agg_column, table_name = agg_segment.groups()
    elif column_segment:
        agg_func, table_name = column_segment.groups()
        agg_column = None
    elif all_segment:
        agg_func = 'ALL'
        table_name = all_segment.group(1)
        agg_column = None
    else:
        print("Invalid FIND format.")
        return

    # print('agg_func is', agg_func)

    group_column = group_segment.group(1) if group_segment else None
    order_type = order_segment.group(1) if order_segment else None
    where_column, where_operator, where_value = where_segment.groups() if where_segment else (None, None, None)

    # Read metadata and data
    df_metadata = pd.read_csv('tables_metadata.csv')
    if table_name not in df_metadata['table_name'].values:
        print(f"Table '{table_name}' does not exist.")
        return

    metadata_row = df_metadata[df_metadata['table_name'] == table_name].iloc[0]
    locations = int(metadata_row['locations']) + 1
    file_directory = os.path.join(os.getcwd(), 'files', table_name)

    # Initialize result containers
    final_result = None
    partial_results = []

    # Process each file separately
    for i in range(locations):
       
        file_path = f"{file_directory}/{table_name}_{i}.csv"
        df = pd.read_csv(file_path, low_memory=False)

        # Apply WHERE clause
        # Apply filtering if WHERE clause is present
        if where_column and where_operator and where_value:
            where_value = convert_data_types(where_value)
            # print('!There is a where_column, where_value = ', where_value, 'where_operator = ', where_operator, 'type = ', type(where_value))
            if where_operator == '=':
                df = df[df[where_column] == where_value]
            elif where_operator == '!=':
                df = df[df[where_column] != where_value]
            elif where_operator == '>':
                df = df[df[where_column] > where_value]
            elif where_operator == '<':
                df = df[df[where_column] < where_value]
        

        # Collect data for non-aggregated query (specific column selection)
        if agg_func and not agg_segment and not all_segment:
            # print('Precise selection!')
            selected_data = df[[agg_func]]
            partial_results.append(selected_data)

        # Collect data for aggregation
        if group_column:
            if agg_func.upper() in ['COUNT', '#']:
                # Count the number of entries for each group
                group_counts = df.groupby(group_column).size()
                partial_results.append(group_counts.reset_index(name='COUNT'))
            elif agg_func.upper() in ['MIN', 'MAX', 'SUM']:
                # Perform aggregation for each group
                agg_func_lower = agg_func.lower()
                grouped_result = df.groupby(group_column)[agg_column].agg(agg_func_lower).reset_index()
                partial_results.append(grouped_result)
        elif agg_func.upper() in ['MIN', 'MAX', 'SUM']:
            agg_func_lower = agg_func.lower()
            if agg_func.upper() == 'SUM':
                sum_result = pd.DataFrame({agg_func: [df[agg_column].sum()]})
                partial_results.append(sum_result)
            elif agg_func.upper() == 'MIN':
                min_result = pd.DataFrame({agg_func: [df[agg_column].min()]})
                partial_results.append(min_result)
            elif agg_func.upper() == 'MAX':
                max_result = pd.DataFrame({agg_func: [df[agg_column].max()]})
                partial_results.append(max_result)
        elif agg_func.upper() in ['COUNT', '#']:
            count_result = pd.DataFrame({'COUNT': [df.shape[0]]})
            partial_results.append(count_result)
        elif agg_func.upper() == 'ALL':
            partial_results.append(df)

        # print('partial:\n', partial_results)
        # Combine results based on query type

    # print('End of loop. partial:\n', partial_results)
    if group_column and agg_func.upper() in ['MIN', 'MAX', 'SUM', 'COUNT', '#']:
        combined_grouped = pd.concat(partial_results)
        final_result = combined_grouped.groupby(group_column).sum().reset_index() if agg_func.upper() in ['COUNT', '#'] else combined_grouped.groupby(group_column).agg(agg_func_lower).reset_index()
    elif agg_func.upper() in ['MIN', 'MAX', 'SUM']:
        final_result = pd.DataFrame({agg_func: [sum(r[agg_func].iloc[0] for r in partial_results)]})
    elif agg_func.upper() in ['COUNT', '#']:
        final_result = pd.DataFrame({'COUNT': [sum(r['COUNT'].iloc[0] for r in partial_results)]})
    elif agg_func.upper() == 'ALL':
        final_result = pd.concat(partial_results)
    else:
        final_result = pd.concat(partial_results)

    # Apply ordering if WITH ORDER is present
    if order_type and final_result is not None:
        ascending = True if order_type == 'ASC' else False
        if group_column and len(final_result.columns) > 1:
            # Sort by the second column (aggregated value) in grouped results
            final_result = final_result.sort_values(by=final_result.columns[1], ascending=ascending)
        elif not group_column:
            # Sort by the first column in ungrouped results
            final_result = final_result.sort_values(by=final_result.columns[0], ascending=ascending)
    
    if final_result is not None:
        print(final_result.to_string(index=False))
        # pass
    else:
        print("No results to display.")

def execute_chunk_join_query(command ):
    metadata_file = 'tables_metadata.csv'
    # Parse the command
    join_pattern = r"JOIN (\w+) WITH (\w+) AS (\w+) ON THAT (\w+)\.(\w+) = (\w+)\.(\w+)"
    match = re.search(join_pattern, command)
    if not match:
        print("Invalid command format.")
        return

    table1, table2, new_table_name, table1_name, column1, table2_name, column2 = match.groups()
    
    # Define file directories
    file_directory1 = os.path.join(os.getcwd(), 'files', table1)
    file_directory2 = os.path.join(os.getcwd(), 'files', table2)

    # Check if new table already exists
    df_metadata = pd.read_csv(metadata_file)
    if new_table_name in df_metadata['table_name'].values:
        print(f"Table '{new_table_name}' already exists.")
        return

    # Read the first table fully into memory
    df_table1 = pd.concat([pd.read_csv(os.path.join(file_directory1, f)) for f in os.listdir(file_directory1) if f.endswith('.csv')])
    
    # Initialize a container for the chunks
    joined_chunks = []

    # Read all files of the second table and perform join operation
    table2_files = [f for f in os.listdir(file_directory2) if f.endswith('.csv')]
    for file in table2_files:
        df_table2 = pd.read_csv(os.path.join(file_directory2, file))
        joined_chunk = pd.merge(df_table1, df_table2, left_on=column1, right_on=column2)
        joined_chunks.append(joined_chunk)

    # Concatenate all joined chunks
    result = pd.concat(joined_chunks)

    # Use split_and_save_csv to save the result into multiple CSVs and update metadata
    result_file_path = os.path.join(os.getcwd(), f"{new_table_name}_temp.csv")
    result.to_csv(result_file_path, index=False)
    split_and_save_csv(f"{new_table_name}_temp.csv", new_table_name)

    print(f"Table '{new_table_name}' created and saved successfully.")

def clear_database():
    current_directory = os.getcwd()
    metadata_file = os.path.join(current_directory, 'tables_metadata.csv')
    files_directory = os.path.join(current_directory, 'files')

    # Delete the metadata file
    try:
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
            print(f"Metadata file '{metadata_file}' has been deleted.")
    except OSError as e:
        print(f"Error: {e.strerror}. Metadata file '{metadata_file}' could not be deleted.")

    # Delete the files directory
    try:
        if os.path.exists(files_directory):
            shutil.rmtree(files_directory)
            print(f"Directory '{files_directory}' and all its contents have been deleted.")
    except OSError as e:
        print(f"Error: {e.strerror}. Directory '{files_directory}' could not be deleted.")

def delete_table(command):
    # Regular expression to extract table name from command
    pattern = r"DELETE TABLE (\w+)"
    match = re.match(pattern, command)
    if not match:
        print("Invalid DELETE TABLE command.")
        return

    table_name = match.group(1)
    file_directory = os.path.join(os.getcwd(), 'files', table_name)

    # Check if table exists
    if os.path.exists(file_directory):
        # Delete table files and metadata
        delete_table_files_and_metadata(table_name, file_directory)
    else:
        # If table directory doesn't exist, just try to delete metadata
        delete_metadata_entry(table_name)

def main():
    system_initiate()
    while True:
        command = input("OurDB > ").strip()
        # EXIT
        if command.upper() == "EXIT" or command.upper() == 'E':
            break

        # HELP to print reference text
        if command.upper() == 'HELP':
            print_help()
            continue
        
        # CLEAR THE DATABASE, REBOOT THE SYSTEM
        if command.upper() == 'CLEAR DATABASE':
            confirm_user = input('The entire database will be deleted. Please confirm action. Yes/No').strip()
            if confirm_user.upper() == 'YES' or confirm_user.upper() == 'Y':
                clear_database()
                break
            continue
        
        # DELETE A SPECIFIC TABLE
        if command.upper().startswith('DELETE TABLE'):
            delete_table(command)
            continue

        # SHOW TABLES to give all information of the database, namely, print the metadata information
        if command.upper() == 'SHOW TABLES':
            print_metadata()
            continue

        # SHOW TABLES NAME will only show the names of available tables in the database
        if command.upper() == 'SHOW TABLES NAME':
            print_tables_name()
            continue
        
        # show several rows of a specific table: SHOW cars LIMIT 10
        if command.upper().startswith('SHOW'): 
            show_table_rows(command)
            continue
        
        # Create new table by input
        # usage: 'CREATE TABLE table_name [name, revenue, employee] WITH VALUES [a, b, c]'.
        if command.upper().startswith('CREATE TABLE'):
            create_table(command)
            continue
            # CREATE TABLE my_table [name, revenue, employee] WITH VALUES [CompanyA, 1000000, 50]

        # Create new table from a existing csv file
        if command.upper().startswith('IMPORT FROM'):
            import_from_file(command)
            continue
            # IMPORT FROM used_car_sale.csv AS car_sale
            # IMPORT FROM car_manufacturers.csv AS manu
        
        # updating method: UPDATE TO TABLE cars LET mileage = mileage - 5 WHERE year = '2025'
        if command.upper().startswith('UPDATE TO TABLE'):
            update_table(command)
            continue

        # inserting method: ADD TO TABLE table_name WITH VALUES [a,b,c,...]
        if command.upper().startswith('ADD TO TABLE'): # command = 'ADD TO TABLE cars WITH VALUES [Ziyu, 2024, 10]'
            insert_query_handle(command)
            continue
            # 'ADD TO TABLE cars WITH VALUES [NewBrand, 2025, 0]'
            # ADD TO TABLE cars WITH VALUES [Audi, 1997, 199700]
            # ADD TO TABLE cars WITH VALUES [Tesla, 1997, 300000]
        
        # deleting method: REMOVE FROM TABLE table_name THAT column_name > value, now support <,>,=,!=
        if command.upper().startswith('REMOVE FROM TABLE'): # command = 'REMOVE FROM TABLE cars THAT year < 2020'
            delete_query_handle(command)
            continue
            # REMOVE FROM TABLE cars THAT model != NewBrand
            # REMOVE FROM TABLE cars THAT mileage > 200000
        
        if command.upper().startswith('FIND'):
            execute_select_query(command)
            continue
        # updating values: UPDATE TO TABLE cars LET mileage = mileage * 1.1 WHERE model = 'Tesla'

        if command.upper().startswith('JOIN'):
            execute_chunk_join_query(command)
            continue

        # if all fail
        print("Unknown command.")

if __name__ == "__main__":
    main()



