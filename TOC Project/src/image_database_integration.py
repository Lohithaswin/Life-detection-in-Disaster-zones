import cx_Oracle

# Oracle database connection details
username = "system"
password = "hello123"
host = "localhost"  # e.g., localhost or cloud instance IP
service_name = "XE"

dsn = cx_Oracle.makedsn(host, port=1521, service_name=service_name)

# Connect to the Oracle database
connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)
cursor = connection.cursor()

# Function to insert image details
def insert_image_details(image_name, image_path):
    query = "INSERT INTO image_details (image_name, image_path) VALUES (:1, :2)"
    cursor.execute(query, [image_name, image_path])
    connection.commit()
    print("Image details inserted successfully!")

# Query image details
def query_image_details():
    cursor.execute("SELECT * FROM image_details")
    for row in cursor.fetchall():
        print(row)

# Close the connection
def close_connection():
    cursor.close()
    connection.close()