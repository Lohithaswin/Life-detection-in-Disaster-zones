import cx_Oracle

# Oracle database connection details
username = "system"
password = "dbms123"
host = "localhost"
port = 1521
service_name = "XE"

dsn = cx_Oracle.makedsn(host, port=port, service_name=service_name)

# Function to insert video details
def insert_video_details(video_name, video_path):
    try:
        connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)
        cursor = connection.cursor()

        query = "INSERT INTO video_details (video_name, video_path) VALUES (:1, :2)"
        cursor.execute(query, [video_name, video_path])
        connection.commit()
        print(f"✅ Inserted video: {video_name}")

    except cx_Oracle.DatabaseError as e:
        print("❌ Database error:", e)

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

# Function to insert image details
def insert_image_details(image_name, image_path):
    try:
        connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)
        cursor = connection.cursor()

        query = "INSERT INTO image_details (image_name, image_path) VALUES (:1, :2)"
        cursor.execute(query, [image_name, image_path])
        connection.commit()
        print(f"✅ Inserted image: {image_name}")

    except cx_Oracle.DatabaseError as e:
        print("❌ Database error:", e)

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()